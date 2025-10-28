import type { ApiProvider, CallApiContextParams, ProviderOptions, ProviderResponse, TokenUsage } from 'promptfoo';
import { runWorkflow } from './workflow.js';

/**
 * Tracking information returned by the workflow
 */
interface WorkflowTracking {
  totalPromptTokens: number;
  totalCompletionTokens: number;
  totalTokens: number;
  totalCost: number;
  agentCalls: Array<{
    agent: string;
    model: string;
    promptTokens: number;
    completionTokens: number;
    totalTokens: number;
    cost: number;
  }>;
}

/**
 * Possible workflow result types
 */
interface WorkflowResult {
  tracking?: WorkflowTracking;
  message?: string;
  output_text?: string;
  classification?: string;
  status?: string;
  jailbreak?: {
    failed: boolean;
    error?: string;
  };
  pii?: {
    failed: boolean;
    detected_counts?: string[];
    error?: string;
  };
  moderation?: {
    failed: boolean;
    flagged_categories?: string[];
    error?: string;
  };
  hallucination?: {
    failed: boolean;
    reasoning?: string;
    hallucination_type?: string;
    hallucinated_statements?: string[];
    verified_statements?: string[];
    error?: string;
  };
}

/**
 * Input format for the workflow
 */
interface WorkflowInput {
  input_as_text?: string;
  messages?: Array<{ role: string; content: string }>;
}

/**
 * Red team provider for OpenAI Agents customer service workflow
 *
 * Features:
 * - Multi-turn conversation support for Crescendo attacks
 * - Cost and token tracking
 * - Guardrail failure detection
 * - Type-safe integration with promptfoo
 */
export default class RedTeamAgentProvider implements ApiProvider {
  protected providerId: string;
  public config: Record<string, unknown>;
  private verbose: boolean;

  constructor(options: ProviderOptions) {
    this.providerId = options.id || 'openai-agents-redteam';
    this.config = options.config || {};
    this.verbose = (this.config.verbose as boolean) ?? false;
  }

  id(): string {
    return this.providerId;
  }

  async callApi(prompt: string, context?: CallApiContextParams): Promise<ProviderResponse> {
    try {
      // Parse input - support both single-turn text and multi-turn message arrays
      const workflowInput = this.parseInput(prompt);

      if (this.verbose) {
        console.log('[RedTeam Provider] Input:', JSON.stringify(workflowInput, null, 2));
      }

      // Execute workflow
      const result = await runWorkflow(workflowInput);

      if (this.verbose) {
        console.log('[RedTeam Provider] Result:', JSON.stringify(result, null, 2));
      }

      // Extract structured response
      const output = this.extractOutput(result);
      const tokenUsage = this.extractTokenUsage(result);
      const cost = result.tracking?.totalCost || 0;
      const metadata = this.extractMetadata(result, context);

      return {
        output,
        tokenUsage,
        cost,
        metadata,
      };
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Workflow execution failed';
      console.error('[RedTeam Provider] Error:', errorMessage);
      return {
        error: errorMessage,
        output: '',
      };
    }
  }

  /**
   * Parse input string into workflow input format
   * Supports both plain text and JSON message arrays for multi-turn conversations
   */
  private parseInput(prompt: string): WorkflowInput {
    try {
      const parsed = JSON.parse(prompt);

      // Check if it's a message array (for multi-turn support)
      if (Array.isArray(parsed) && parsed.length > 0 && parsed[0].role) {
        if (this.verbose) {
          console.log('[RedTeam Provider] Multi-turn conversation detected:', parsed.length, 'messages');
        }
        return { messages: parsed };
      }
    } catch (e) {
      // Not JSON, treat as plain text
    }

    // Fallback to single-turn text
    return { input_as_text: prompt };
  }

  /**
   * Extract output text from various result formats
   * Handles guardrail failures with informative messages
   */
  private extractOutput(result: WorkflowResult): string {
    // Handle guardrail failures
    if (result.jailbreak?.failed) {
      return 'Request blocked by guardrails: Jailbreak attempt detected';
    }
    if (result.pii?.failed) {
      const counts = result.pii.detected_counts?.join(', ') || 'unknown';
      return `Request blocked by guardrails: PII detected (${counts})`;
    }
    if (result.moderation?.failed) {
      const categories = result.moderation.flagged_categories?.join(', ') || 'unknown';
      return `Request blocked by guardrails: Content policy violation (${categories})`;
    }
    if (result.hallucination?.failed) {
      return 'Request blocked by guardrails: Hallucination detected';
    }

    // Handle successful responses
    if (result.message) {
      return result.message;
    }
    if (result.output_text) {
      return result.output_text;
    }

    // Fallback
    return JSON.stringify(result);
  }

  /**
   * Extract token usage in promptfoo format
   */
  private extractTokenUsage(result: WorkflowResult): TokenUsage | undefined {
    if (!result.tracking) {
      return undefined;
    }

    return {
      total: result.tracking.totalTokens,
      prompt: result.tracking.totalPromptTokens,
      completion: result.tracking.totalCompletionTokens,
    };
  }

  /**
   * Extract metadata including agent calls, classification, and guardrail info
   */
  private extractMetadata(result: WorkflowResult, context?: CallApiContextParams): Record<string, unknown> {
    const metadata: Record<string, unknown> = {};

    // Add agent call breakdown (for cost analysis)
    if (result.tracking?.agentCalls) {
      metadata.agentCalls = result.tracking.agentCalls;
    }

    // Add classification result
    if (result.classification) {
      metadata.classification = result.classification;
    }

    // Add workflow status
    if (result.status) {
      metadata.status = result.status;
    }

    // Add guardrail failures (for red team analysis)
    const guardrailFailures: Record<string, unknown> = {};
    let hasFailures = false;

    if (result.jailbreak) {
      guardrailFailures.jailbreak = result.jailbreak;
      hasFailures = true;
    }
    if (result.pii) {
      guardrailFailures.pii = result.pii;
      hasFailures = true;
    }
    if (result.moderation) {
      guardrailFailures.moderation = result.moderation;
      hasFailures = true;
    }
    if (result.hallucination) {
      guardrailFailures.hallucination = result.hallucination;
      hasFailures = true;
    }

    if (hasFailures) {
      metadata.guardrail_failure = guardrailFailures;
    }

    // Pass through test context variables (for red team correlation)
    if (context?.vars) {
      metadata.test_vars = context.vars;
    }

    return metadata;
  }
}
