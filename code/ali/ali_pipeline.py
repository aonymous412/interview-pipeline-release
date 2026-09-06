"""ALI orchestrator -- the five-component elicitation pipeline and its formal
stopping criterion: while cov(F) < theta_d and n_turns < 15.

ALI is the framework the paper proposes, instantiated over three deployments.
What runs in *this* evaluation harness differs from what runs in production,
and the difference is deliberate -- see README.md and PROVENANCE.md:

  - ALI[Rad-Assist] (Medical). The production system is not a native C0-C4
    pipeline: it is a hybrid Mistral 7B/GRPO extractor that does not follow the
    decomposition. The paper states this. For the evaluation, C0-C4 all run on
    the hosted model, so the Medical numbers measure the ALI decomposition and
    not the deployed clinical extractor.
  - ALI[Telos] (Consulting). C1/C4 are GPT-2 + LoRA fine-tunes.
  - ALI[FinAgent] (Payments). C1 = Qwen2.5-0.5B + LoRA, C4 = Qwen2.5-1.5B + LoRA;
    the evaluated ALI_v1 falls back to the hosted model for C4.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from . import config
from .coverage_scorer import CoverageScorer, load_taxonomy
from .components.c0_detector import C0Detector
from .components.c1_identifier import C1Identifier
from .components.c2_clusterer import C2Clusterer
from .components.c3_generator import C3Generator
from .components.c4_extractor import C4Extractor

# Per-deployment stopping thresholds.
#
# These are the values that PRODUCED the published records, and they are the
# values the paper reports (Consulting 0.90, Medical 0.90, Payments 0.75).
#
# Do not re-align them with the paper's description of the *deployed* systems.
# The deployed clinical configuration is stricter than the evaluated one -- an
# unresolved high-weight parameter (w >= 80) can corrupt the downstream
# prognostic model -- and the paper reports the deployed range separately
# (0.85-0.95, scaled by domain) from the evaluated theta of 0.90. Editing these
# constants to match the deployment description would make the code stop
# reproducing the shipped results. Of the 46 Medical runs that halt on
# coverage_reached, 20 do so at cov = 0.901 and 22 finish below 0.92, so a
# stricter threshold would not have stopped them where the records say it did.
# (verify_numbers.py re-derives these counts from results/ali/.)
THETA = {
    "telos": 0.90,       # dynamique 0.85-0.95 selon nb d'elements ; 0.90 = default COVERAGE_THRESHOLD
    "finagent": 0.75,  # COVERAGE_THRESHOLD fixe (conversation_loop.py)
    "rad_assist": 0.90,  # = seuil du run publie (cf. avertissement ci-dessus)
}


@dataclass
class ConversationState:
    domain: str
    prompt: str
    taxonomy: dict
    prompt_id: str = ""
    F: dict = field(default_factory=dict)
    history: list = field(default_factory=list)
    mentioned: set = field(default_factory=set)
    cov_history: list = field(default_factory=list)
    stopped_by: str | None = None
    native_domain: str | None = None  # FinAgent uniquement


class ALIPipeline:
    def __init__(self, deployment: str, c1_mode: str = "auto", c4_mode: str = "auto"):
        """
        deployment : "rad_assist" | "telos" | "finagent"
        c1_mode / c4_mode : "finetuned" | "gemini" | "auto"
        """
        self.deployment = deployment
        d = config.DEPLOYMENT_DEFAULTS[deployment]
        self.taxonomy = load_taxonomy(d["domain_key"])
        self.theta = THETA[deployment]
        self.c0 = C0Detector()
        self.c1 = C1Identifier(deployment, mode=c1_mode)
        self.c2 = C2Clusterer()
        self.c3 = C3Generator()
        self.c4 = C4Extractor(deployment, mode=c4_mode)
        self.scorer = CoverageScorer(self.taxonomy)

    def run(self, initial_prompt: str, user_fn, prompt_id: str = "") -> dict:
        state = ConversationState(
            domain=self.deployment,
            prompt=initial_prompt,
            taxonomy=self.taxonomy,
            prompt_id=prompt_id,
        )

        if self.deployment == "finagent":
            state.native_domain = self.c0.detect_finagent_native_domain(initial_prompt)

        state.F = self.c0.extract(initial_prompt, self.taxonomy)
        state.cov_history.append(self.scorer.compute(state.F))

        while self.scorer.compute(state.F) < self.theta:
            if len(state.history) >= config.MAX_TURNS:
                state.stopped_by = "turn_cap"
                break

            missing = self.c1.identify(state)
            if not missing:
                state.stopped_by = "c1_empty"
                break

            clusters = self.c2.cluster(missing)
            if not clusters:
                state.stopped_by = "c2_empty"
                break

            question = self.c3.generate(clusters[0], state)
            response = user_fn(question)

            state.history.append((question, response))
            self.c4.extract(response, missing, state)

        if state.stopped_by is None:
            state.stopped_by = "coverage_reached"

        return {
            "prompt_id": state.prompt_id,
            "deployment": self.deployment,
            "native_domain": state.native_domain,
            "final_cov": self.scorer.compute(state.F),
            "n_turns": len(state.history),
            "stopped_by": state.stopped_by,
            "F_resolved": state.F,
            "transcript": state.history,
            "cov_by_turn": state.cov_history,
        }
