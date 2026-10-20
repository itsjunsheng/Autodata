# Reading notes: Autodata → visual grounding

Read source: all 28 pages of `Autodata.pdf`, including Figures 1–6, experiments, discussion, references, and Appendices A–C. The supplied version is *Autodata: An agentic data scientist to create high quality synthetic data*, by Kulikov and colleagues; the first page lists July 7, 2026 and arXiv:2606.25996v3.

## What the paper proposes

**Section 2 / Figure 1 (pages 2–3):** the agent acts as a data scientist. It grounds creation in source material, inspects generated data, measures performance, derives learnings, and revises the generation recipe. The product is training or evaluation data. Simply asking a VLM to predict a box, even repeatedly, would not implement this data-creation objective.

**Section 2.1 / Figure 2 (pages 3–4):** Agentic Self-Instruct instantiates this with a challenger, weak solver, strong solver, and verifier/judge under an orchestrator. Solver behaviour and quality feedback shape later candidates. The strong solver helps establish answerability, while the weak solver identifies opportunities for the learner. The paper allows the same base model in different modes, but that is not necessary here: the weak solver is the Qwen3-VL checkpoint we intend to train.

**CS experiments (pages 4–6):** initial examples are too easy, so the loop searches for meaningful strong–weak separation. Quality checks catch context leakage and weak rubrics. The accepted data improves downstream RL results relative to the paper's prompted baseline. Repeated solver calls help reduce noise.

**Legal experiments (pages 6–9):** initial examples are too difficult, with many weak rollouts scoring zero. The loop improves the usefulness of the reward distribution rather than blindly maximising difficulty or the solver gap. A structured judge analyses per-rollout patterns and sends concrete feedback. This is why “make Qwen fail” alone is not a sufficient definition of high quality.

**Scientific reasoning (pages 9–10):** generated mathematical-object problems improve average held-out results, but pass-at-k results are more mixed. Appendix A (pages 19–20) attributes a substantial part of improvement to fewer truncated model responses as well as improved reasoning. Appendix B (pages 20–21) analyses the distribution of reasoning, knowledge, and mixed task types. These results concern those tasks and training setups, not visual grounding.

**Meta-optimisation (pages 11–12 / Figure 6):** an outer process analyses trajectories, edits the agent scaffold/prompt, reevaluates on held-out material, and retains improvements. This is distinct from revising one example within the inner loop. This implementation contains only the inner loop.

**Related work and discussion (pages 13–15):** the distinguishing feature is adapting creation based on measured usefulness, not just static synthetic generation and filtering. The authors discuss reward hacking, examples that are difficult for unhelpful reasons, dataset-level analysis, and human co-research. In particular, making a weak model intentionally behave poorly is not a legitimate difficulty signal.

**Appendix C (pages 21–28):** the role prompts require actual solver evaluation before a candidate is accepted, bounded improvement loops, diagnostic feedback, and incremental persistence. The CS prompt explicitly distinguishes infrastructure failures from poor examples. The scientific-reasoning prompt uses repeated solver outcomes, not a guess that a task will separate models.

The paper's settings vary by experiment: Section 3.1, the meta-optimisation setup, and the appendix do not all use the same numeric accept conditions. The evolved CS prompt also differs from the earlier rubric prompts on negative weights. We therefore define and document a separate, fixed three-check rule for this task instead of claiming one universal paper threshold.

## Mapping to this implementation

| Paper component | Localisation implementation |
| --- | --- |
| Grounding source | Original RefCOCO train image, target annotation, source referring expressions |
| Main agent | OpenAI LLM creates the challenger prompt, analyses previous outcomes, and revises the strategy (`main_agent.py`) |
| Challenger | OpenAI VLM creates a referring expression from the main agent's prompt (`challenger.py`) |
| Referring-expression checks | Candidates go directly to both solvers; a VLM judge (`judge.py`) then reviews each round that passes the three IoU checks for visible truth of every claim and for other matching objects |
| Weak solver | The target Qwen3-VL model called through OpenRouter (`solver.py`) |
| Strong solver | Qwen3-VL-235B on OpenRouter by default, or an OpenAI VLM; correctness measured per candidate (`solver.py`) |
| Objective evaluator | Python IoU against the unchanged RefCOCO annotation |
| Verifier implementation | Deterministic IoU calculation and threshold comparison, with no model calls (`verifier.py`) |
| Execution harness | Python enforces label boundaries, fixed gates, and bounded rounds around the main LLM |
| Learnings | Main LLM analyses solver boxes, IoUs, and deterministic rejection reasons to issue a new challenger prompt |
| Final data | Original image, accepted generated referring expression, annotated box in Qwen3-VL format |
| Trajectories | Supabase stores one row per round, with accepted rounds forming the training set; Google Drive stores one coloured image per completed/failed round |

The implementation imports every referred object of each image and uses one candidate per round, one rollout per solver, at most six rounds, and at most one accepted example per input sample. Each diagram role has a module; the two solvers share one class. Five roles make model calls; the verifier is deterministic Python. The main agent performs actual LLM planning/reflection; a small Python harness enforces the workflow and the verifier's gates. There is no agent framework, code execution tool, or autonomous edit to scoring code. This is the inner data-creation loop, not the paper's outer meta-optimisation procedure.

## Quality boundaries

- The answer is fixed before generation. The main agent and challenger cannot change the ground-truth box or the verifier thresholds.
- Annotated images and target metadata go only to the main agent and challenger; the deterministic verifier receives ground-truth and predicted boxes. Solvers see clean images and referring expressions in fresh requests, without main-agent instructions, with no system turn, and in the same user prompt as the training file.
- Both solvers receive the same task and coordinate convention. Qwen is never instructed to be weak.
- Every generated referring expression is sent to the solvers, including duplicates. The judge runs only after the three checks pass, never before the solvers.
- Solver agreement or a high IoU does not establish referring expression truthfulness or uniqueness; the judge checks both on passing rounds and is itself a model judgment with its own error rate.
- Provider failures, refusals, truncation, and invalid coordinates produce an error, not a successful weak-failure criterion.
- The final answer comes from RefCOCO, not a potentially imprecise model prediction.
- Rejected/exhausted targets remain inspectable and are excluded from the training bundle.
- Generation uses training samples only; the source adapter enforces training-only without a stored split attribute. The ten examples are a workflow pilot, not an evaluation benchmark.

## Accept rule

A round is accepted only when strong IoU ≥ 0.5, weak IoU < 0.5, and the IoU gap is ≥ 0.2. This is an initial binary-localisation adaptation of the paper's weak/strong setup. It cannot establish GRPO suitability from one rollout, and it does not guarantee better supervised training. The judge adds a semantic check on passing rounds; it does not make the rule's thresholds any less fixed.

Strong/weak names are roles, not measured model rankings. A general OpenAI VLM may underperform Qwen on precise box placement. Such cases should be diagnosed in traces and addressed through solver selection or an improved localisation harness; the code does not silently relax correctness.

This is an intentional simplification of the paper's verifier/judge: generated referring expressions are retained through solver comprehension and deterministic geometry, with a VLM judge reviewing only the rounds that pass. Historical runs used the earlier quality gate; their records are not reevaluated automatically.

## Later experiments

Once the ten-example loop runs with live endpoints, inspect retained and rejected cases, evaluate the solver gap over repeated rounds, measure diversity, and compare Qwen trained on equal-sized baseline and curated datasets using untouched RefCOCO evaluation splits. Dataset-level strategy learning, training the target model between rounds, and outer scaffold meta-optimisation can follow after that evidence exists.
