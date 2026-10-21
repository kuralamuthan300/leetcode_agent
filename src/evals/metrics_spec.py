"""Frozen metric contract for Eval Framework v1.

This module is the single source of truth for every number the scorecard emits.
``evals.report`` builds its tables from :data:`METRICS`; ``evals.scorers`` builds
its ``@scorer`` functions from the entries whose scope is ``per_problem``. A
metric therefore cannot be renamed, redefined, or given a different denominator
in one place without the other place changing too.

The human-readable view of the same contract lives in ``plan/eval_plan.md``
(section 1, "Recommended v1 scorecard"). Keep the two in sync: tests read the
plan table and assert that every primary metric ID is present there.

Scorecard rows 4, 9, and 10 contain more than one reported number. Those extra
numbers are still first-class specs here (``primary=False``) so that every
number in ``metrics.json`` has an explicit numerator, denominator, and gate.
The ten ``primary`` specs are the ten rows of ``scorecard.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

__all__ = [
    "CANONICAL_IDS",
    "FEEDBACKS",
    "METRICS",
    "METRICS_BY_MLFLOW_KEY",
    "PRIMARY_METRICS",
    "SCORERS",
    "Aggregation",
    "Artifact",
    "Direction",
    "MetricSpec",
    "Scope",
    "Tier",
    "blocking_metrics",
    "feedback_specs",
    "get_metric",
    "metric_ids_for_row",
    "per_problem_metrics",
    "primary_metrics",
    "run_level_metrics",
    "scorer_specs",
]


class Tier(str, Enum):
    """Which question a metric answers (the scorecard's four sections)."""

    OUTCOME = "outcome"
    CALIBRATION = "calibration"
    TESTS = "tests"
    OPS = "ops"


class Direction(str, Enum):
    """The direction a healthy metric moves in."""

    HIGHER = "higher_is_better"
    LOWER = "lower_is_better"


class Scope(str, Enum):
    """Whether MLflow sees the metric per problem or once per eval run."""

    PER_PROBLEM = "per_problem"
    RUN = "run"


class Aggregation(str, Enum):
    """Aggregation used by the matching ``mlflow.genai`` scorer.

    ``none`` means the metric is logged with ``mlflow.log_metrics`` and has no
    per-problem scorer.
    """

    MEAN = "mean"
    MIN = "min"
    P90 = "p90"
    NONE = "none"


class Artifact(str, Enum):
    """Inputs a metric needs before it can be computed."""

    RUN_JSON = "run_json"
    GRADE_JSON = "grade_json"
    REFERENCE = "reference"
    ORACLE_JSON = "oracle_json"
    MUTATION_JSON = "mutation_json"
    CONSTRAINTS_JSON = "constraints_json"
    RUN_INDEX = "run_index"
    MANIFEST = "manifest"
    TRACE = "trace"


@dataclass(frozen=True, slots=True)
class MetricSpec:
    """One frozen definition of one scorecard number.

    Attributes are deliberately flat (``gate_value`` / ``blocking`` rather than a
    nested gate object) so they can be written verbatim to ``metrics.json`` and
    ``manifest.json`` without a second mapping layer.
    """

    id: str
    display_name: str
    scorecard_row: int
    tier: Tier
    numerator: str
    denominator: str
    direction: Direction
    gate_value: float | None
    blocking: bool
    required_artifacts: tuple[Artifact, ...]
    mlflow_metric_key: str
    mlflow_scorer_name: str | None
    mlflow_feedback_name: str | None
    scope: Scope
    aggregation: Aggregation
    primary: bool = False
    unit: str = "ratio"
    baseline_margin: float | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.id or not self.display_name:
            raise ValueError("metric id and display_name must be non-empty")
        if self.scorecard_row not in range(1, 11):
            raise ValueError(f"{self.id}: scorecard_row must be between 1 and 10")
        if not self.numerator or not self.denominator:
            raise ValueError(f"{self.id}: numerator and denominator are required")
        if self.blocking and self.gate_value is None:
            raise ValueError(f"{self.id}: a blocking metric needs a gate_value")
        if not self.required_artifacts:
            raise ValueError(f"{self.id}: at least one required artifact is needed")

        if self.scope is Scope.PER_PROBLEM:
            if not self.mlflow_scorer_name or not self.mlflow_feedback_name:
                raise ValueError(
                    f"{self.id}: per-problem metrics need a scorer name and feedback name"
                )
            if self.aggregation is Aggregation.NONE:
                raise ValueError(f"{self.id}: per-problem metrics need a scorer aggregation")
        else:
            if self.mlflow_scorer_name is not None or self.mlflow_feedback_name is not None:
                raise ValueError(
                    f"{self.id}: run-level metrics cannot declare a scorer or trace feedback"
                )
            if self.aggregation is not Aggregation.NONE:
                raise ValueError(f"{self.id}: run-level metrics must use aggregation='none'")

    @property
    def is_per_problem(self) -> bool:
        return self.scope is Scope.PER_PROBLEM

    @property
    def is_run_level(self) -> bool:
        return self.scope is Scope.RUN

    @property
    def gate_operator(self) -> str | None:
        """Comparison used to evaluate the gate, or ``None`` when report-only."""
        if self.gate_value is None:
            return None
        return ">=" if self.direction is Direction.HIGHER else "<="

    @property
    def scorer_aggregation(self) -> str:
        return self.aggregation.value

    def to_dict(self) -> dict[str, object]:
        """Primitive-only representation for manifests and ``metrics.json``."""
        return {
            "id": self.id,
            "display_name": self.display_name,
            "scorecard_row": self.scorecard_row,
            "tier": self.tier.value,
            "numerator": self.numerator,
            "denominator": self.denominator,
            "direction": self.direction.value,
            "gate_value": self.gate_value,
            "gate_operator": self.gate_operator,
            "blocking": self.blocking,
            "required_artifacts": [artifact.value for artifact in self.required_artifacts],
            "mlflow_metric_key": self.mlflow_metric_key,
            "mlflow_scorer_name": self.mlflow_scorer_name,
            "mlflow_feedback_name": self.mlflow_feedback_name,
            "scope": self.scope.value,
            "aggregation": self.aggregation.value,
            "primary": self.primary,
            "unit": self.unit,
            "baseline_margin": self.baseline_margin,
            "notes": self.notes,
        }


# Common artifact groups, kept short so the metric list stays readable.
_OUTCOME = (Artifact.RUN_JSON, Artifact.GRADE_JSON, Artifact.REFERENCE)
_CALIBRATION = (Artifact.RUN_JSON, Artifact.GRADE_JSON)


_FIXTURES: tuple[MetricSpec, ...] = (
    # ------------------------------------------------------------------ row 1
    MetricSpec(
        id="golden_strict_solve_rate",
        display_name="Golden Strict Solve Rate",
        scorecard_row=1,
        tier=Tier.OUTCOME,
        numerator="Problems whose base, edge, and adversarial golden cases all pass",
        denominator="Problems evaluated in the batch (50 for a full run)",
        direction=Direction.HIGHER,
        gate_value=0.70,
        blocking=True,
        required_artifacts=_OUTCOME,
        mlflow_metric_key="golden_strict_solve_rate",
        mlflow_scorer_name="golden_strict_pass",
        mlflow_feedback_name="golden_strict_pass",
        scope=Scope.PER_PROBLEM,
        aggregation=Aggregation.MEAN,
        primary=True,
        baseline_margin=0.05,
        notes=(
            "Hard floor becomes baseline minus 5 points in Step 14; 0.70 is the "
            "initial target before a baseline exists."
        ),
    ),
    MetricSpec(
        id="pass_at_1",
        display_name="Pass@1 (final verdict)",
        scorecard_row=1,
        tier=Tier.OUTCOME,
        numerator="Problems whose final internal verdict is pass",
        denominator="Problems evaluated in the batch",
        direction=Direction.HIGHER,
        gate_value=None,
        blocking=False,
        required_artifacts=_OUTCOME,
        mlflow_metric_key="pass_at_1",
        mlflow_scorer_name=None,
        mlflow_feedback_name=None,
        scope=Scope.RUN,
        aggregation=Aggregation.NONE,
        notes="Final verdict after the retry budget; compare with pass_at_2.",
    ),
    MetricSpec(
        id="pass_at_2",
        display_name="Pass@2 (retry-aware)",
        scorecard_row=1,
        tier=Tier.OUTCOME,
        numerator="Problems that pass at any point within the retry budget",
        denominator="Problems evaluated in the batch",
        direction=Direction.HIGHER,
        gate_value=None,
        blocking=False,
        required_artifacts=_OUTCOME,
        mlflow_metric_key="pass_at_2",
        mlflow_scorer_name=None,
        mlflow_feedback_name=None,
        scope=Scope.RUN,
        aggregation=Aggregation.NONE,
        notes="Uses iterations/eval history; a problem that passes on retry counts once.",
    ),
    MetricSpec(
        id="category_balanced_solve_rate",
        display_name="Category-Balanced Solve Rate",
        scorecard_row=1,
        tier=Tier.OUTCOME,
        numerator="Mean of per-category strict solve rates",
        denominator="Categories represented in the evaluated batch",
        direction=Direction.HIGHER,
        gate_value=None,
        blocking=False,
        required_artifacts=(Artifact.RUN_JSON, Artifact.GRADE_JSON, Artifact.MANIFEST),
        mlflow_metric_key="category_balanced_solve_rate",
        mlflow_scorer_name=None,
        mlflow_feedback_name=None,
        scope=Scope.RUN,
        aggregation=Aggregation.NONE,
        notes="Macro average over categories; stops denser categories from dominating.",
    ),
    # ------------------------------------------------------------------ row 2
    MetricSpec(
        id="golden_case_pass_rate",
        display_name="Golden Case Pass Rate",
        scorecard_row=2,
        tier=Tier.OUTCOME,
        numerator="Golden cases passed (base + edge + adversarial)",
        denominator="Golden cases evaluated (three per problem)",
        direction=Direction.HIGHER,
        gate_value=None,
        blocking=False,
        required_artifacts=_OUTCOME,
        mlflow_metric_key="golden_case_pass_rate",
        mlflow_scorer_name="golden_case_pass_rate",
        mlflow_feedback_name="golden_case_pass_rate",
        scope=Scope.PER_PROBLEM,
        aggregation=Aggregation.MEAN,
        primary=True,
        notes="Overall rate; split into the three case-type specs below.",
    ),
    MetricSpec(
        id="golden_base_case_pass_rate",
        display_name="Golden Base Case Pass Rate",
        scorecard_row=2,
        tier=Tier.OUTCOME,
        numerator="Base golden cases passed",
        denominator="Base golden cases evaluated",
        direction=Direction.HIGHER,
        gate_value=None,
        blocking=False,
        required_artifacts=_OUTCOME,
        mlflow_metric_key="golden_base_case_pass_rate",
        mlflow_scorer_name=None,
        mlflow_feedback_name=None,
        scope=Scope.RUN,
        aggregation=Aggregation.NONE,
    ),
    MetricSpec(
        id="golden_edge_case_pass_rate",
        display_name="Golden Edge Case Pass Rate",
        scorecard_row=2,
        tier=Tier.OUTCOME,
        numerator="Edge golden cases passed",
        denominator="Edge golden cases evaluated",
        direction=Direction.HIGHER,
        gate_value=None,
        blocking=False,
        required_artifacts=_OUTCOME,
        mlflow_metric_key="golden_edge_case_pass_rate",
        mlflow_scorer_name=None,
        mlflow_feedback_name=None,
        scope=Scope.RUN,
        aggregation=Aggregation.NONE,
    ),
    MetricSpec(
        id="golden_adversarial_case_pass_rate",
        display_name="Golden Adversarial Case Pass Rate",
        scorecard_row=2,
        tier=Tier.OUTCOME,
        numerator="Adversarial golden cases passed",
        denominator="Adversarial golden cases evaluated",
        direction=Direction.HIGHER,
        gate_value=0.60,
        blocking=True,
        required_artifacts=_OUTCOME,
        mlflow_metric_key="golden_adversarial_case_pass_rate",
        mlflow_scorer_name=None,
        mlflow_feedback_name=None,
        scope=Scope.RUN,
        aggregation=Aggregation.NONE,
        notes="Hard gate even though the overall case rate is report-only.",
    ),
    # ------------------------------------------------------------------ row 3
    MetricSpec(
        id="false_pass_rate",
        display_name="False-Pass Rate",
        scorecard_row=3,
        tier=Tier.CALIBRATION,
        numerator="False passes: problems the agent marked as passing whose golden verdict fails",
        denominator="Problems the agent marked as passing (internal_pass)",
        direction=Direction.LOWER,
        gate_value=0.05,
        blocking=True,
        required_artifacts=_CALIBRATION,
        mlflow_metric_key="false_pass_rate",
        mlflow_scorer_name="false_pass",
        mlflow_feedback_name="false_pass",
        scope=Scope.PER_PROBLEM,
        aggregation=Aggregation.MEAN,
        primary=True,
        notes=(
            "Conditional risk: when the agent says pass, how often is it wrong? "
            "The scorer's mean is false passes / all problems; metrics.json uses "
            "the conditional denominator above. Report false_pass_share too."
        ),
    ),
    MetricSpec(
        id="false_pass_share",
        display_name="False-Pass Share",
        scorecard_row=3,
        tier=Tier.CALIBRATION,
        numerator="False passes: internal pass but golden fail",
        denominator="Problems evaluated in the batch",
        direction=Direction.LOWER,
        gate_value=None,
        blocking=False,
        required_artifacts=_CALIBRATION,
        mlflow_metric_key="false_pass_share",
        mlflow_scorer_name=None,
        mlflow_feedback_name=None,
        scope=Scope.RUN,
        aggregation=Aggregation.NONE,
        notes="FP / 50 on a full run; reported next to the conditional false-pass rate.",
    ),
    # ------------------------------------------------------------------ row 4
    MetricSpec(
        id="false_fail_rate",
        display_name="False-Fail Rate",
        scorecard_row=4,
        tier=Tier.CALIBRATION,
        numerator="False fails: problems the agent marked as failing whose golden verdict passes",
        denominator="Problems the agent marked as failing (not internal_pass)",
        direction=Direction.LOWER,
        gate_value=0.10,
        blocking=True,
        required_artifacts=_CALIBRATION,
        mlflow_metric_key="false_fail_rate",
        mlflow_scorer_name="false_fail",
        mlflow_feedback_name="false_fail",
        scope=Scope.PER_PROBLEM,
        aggregation=Aggregation.MEAN,
        primary=True,
        notes=(
            "Conditional risk: when the agent says fail, how often is it wrong? "
            "The scorer's mean is false fails / all problems."
        ),
    ),
    MetricSpec(
        id="verdict_agreement",
        display_name="Verdict Agreement",
        scorecard_row=4,
        tier=Tier.CALIBRATION,
        numerator="Problems where internal_pass equals golden_pass (TP + TN)",
        denominator="Problems evaluated in the batch",
        direction=Direction.HIGHER,
        gate_value=None,
        blocking=False,
        required_artifacts=_CALIBRATION,
        mlflow_metric_key="verdict_agreement",
        mlflow_scorer_name=None,
        mlflow_feedback_name=None,
        scope=Scope.RUN,
        aggregation=Aggregation.NONE,
    ),
    MetricSpec(
        id="cohens_kappa",
        display_name="Cohen's Kappa",
        scorecard_row=4,
        tier=Tier.CALIBRATION,
        numerator="Observed agreement minus expected agreement",
        denominator="One minus expected agreement",
        direction=Direction.HIGHER,
        gate_value=0.70,
        blocking=True,
        required_artifacts=_CALIBRATION,
        mlflow_metric_key="cohens_kappa",
        mlflow_scorer_name=None,
        mlflow_feedback_name=None,
        scope=Scope.RUN,
        aggregation=Aggregation.NONE,
        unit="score",
        notes="Chance-corrected agreement; computed from the 2x2 confusion matrix.",
    ),
    MetricSpec(
        id="internal_pass_rate",
        display_name="Internal Pass Rate",
        scorecard_row=4,
        tier=Tier.CALIBRATION,
        numerator="Problems the agent marked as passing (internal_pass)",
        denominator="Problems evaluated in the batch",
        direction=Direction.HIGHER,
        gate_value=None,
        blocking=False,
        required_artifacts=_CALIBRATION,
        mlflow_metric_key="internal_pass_rate",
        mlflow_scorer_name="internal_pass",
        mlflow_feedback_name="internal_pass",
        scope=Scope.PER_PROBLEM,
        aggregation=Aggregation.MEAN,
        notes="Calibration denominator; exposed per trace for filtering.",
    ),
    # ------------------------------------------------------------------ row 5
    MetricSpec(
        id="mutation_score",
        display_name="Mutation Score",
        scorecard_row=5,
        tier=Tier.TESTS,
        numerator="Killed mutants",
        denominator="Killable mutants (reference differs from at least one golden case)",
        direction=Direction.HIGHER,
        gate_value=0.60,
        blocking=True,
        required_artifacts=(
            Artifact.REFERENCE,
            Artifact.MUTATION_JSON,
            Artifact.ORACLE_JSON,
            Artifact.RUN_JSON,
        ),
        mlflow_metric_key="mutation_score",
        mlflow_scorer_name="mutation_score",
        mlflow_feedback_name="mutation_score",
        scope=Scope.PER_PROBLEM,
        aggregation=Aggregation.MEAN,
        primary=True,
        notes=(
            "Canonical scorecard value is the micro score (killed / killable). "
            "The scorer mean is the macro score and is also logged as "
            "mutation_score_macro. Surviving mutants are listed per problem."
        ),
    ),
    MetricSpec(
        id="mutation_score_macro",
        display_name="Mutation Score (Macro)",
        scorecard_row=5,
        tier=Tier.TESTS,
        numerator="Mean of per-problem mutation scores",
        denominator="Problems with at least one killable mutant",
        direction=Direction.HIGHER,
        gate_value=None,
        blocking=False,
        required_artifacts=(Artifact.REFERENCE, Artifact.MUTATION_JSON, Artifact.RUN_JSON),
        mlflow_metric_key="mutation_score_macro",
        mlflow_scorer_name=None,
        mlflow_feedback_name=None,
        scope=Scope.RUN,
        aggregation=Aggregation.NONE,
        notes="Each problem contributes equally, so problems with many mutants do not dominate.",
    ),
    # ------------------------------------------------------------------ row 6
    MetricSpec(
        id="oracle_correctness",
        display_name="Test Oracle Correctness",
        scorecard_row=6,
        tier=Tier.TESTS,
        numerator="Generated tests whose expected output matches the reference solution",
        denominator="Generated tests evaluated (visible + holdout)",
        direction=Direction.HIGHER,
        gate_value=0.95,
        blocking=True,
        required_artifacts=(Artifact.RUN_JSON, Artifact.REFERENCE, Artifact.ORACLE_JSON),
        mlflow_metric_key="oracle_correctness",
        mlflow_scorer_name="oracle_correctness",
        mlflow_feedback_name="oracle_correctness",
        scope=Scope.PER_PROBLEM,
        aggregation=Aggregation.MEAN,
        primary=True,
        notes=(
            "Canonical value is micro over all generated tests; the scorer mean is "
            "the per-problem-balanced view. Split into visible/holdout below."
        ),
    ),
    MetricSpec(
        id="oracle_correctness_agent_tests",
        display_name="Oracle Correctness (Visible Tests)",
        scorecard_row=6,
        tier=Tier.TESTS,
        numerator="Visible generated tests whose expected output matches the reference",
        denominator="Visible generated tests evaluated",
        direction=Direction.HIGHER,
        gate_value=None,
        blocking=False,
        required_artifacts=(Artifact.RUN_JSON, Artifact.REFERENCE, Artifact.ORACLE_JSON),
        mlflow_metric_key="oracle_correctness_agent_tests",
        mlflow_scorer_name=None,
        mlflow_feedback_name=None,
        scope=Scope.RUN,
        aggregation=Aggregation.NONE,
    ),
    MetricSpec(
        id="oracle_correctness_holdout_tests",
        display_name="Oracle Correctness (Holdout Tests)",
        scorecard_row=6,
        tier=Tier.TESTS,
        numerator="Holdout generated tests whose expected output matches the reference",
        denominator="Holdout generated tests evaluated",
        direction=Direction.HIGHER,
        gate_value=None,
        blocking=False,
        required_artifacts=(Artifact.RUN_JSON, Artifact.REFERENCE, Artifact.ORACLE_JSON),
        mlflow_metric_key="oracle_correctness_holdout_tests",
        mlflow_scorer_name=None,
        mlflow_feedback_name=None,
        scope=Scope.RUN,
        aggregation=Aggregation.NONE,
        notes="Holdout tests are coder-hidden, so this isolates testgen quality.",
    ),
    # ------------------------------------------------------------------ row 7
    MetricSpec(
        id="blame_accuracy",
        display_name="Blame Accuracy",
        scorecard_row=7,
        tier=Tier.TESTS,
        numerator="Failures whose culprit (code or test) matches oracle ground truth",
        denominator="Failures with derivable oracle ground truth",
        direction=Direction.HIGHER,
        gate_value=0.80,
        blocking=True,
        required_artifacts=(Artifact.RUN_JSON, Artifact.REFERENCE, Artifact.ORACLE_JSON),
        mlflow_metric_key="blame_accuracy",
        mlflow_scorer_name="blame_accuracy",
        mlflow_feedback_name="blame_accuracy",
        scope=Scope.PER_PROBLEM,
        aggregation=Aggregation.MEAN,
        primary=True,
        notes=(
            "Ground truth: oracle_ok => blame code; otherwise blame test. Repaired "
            "expected outputs use the repaired oracle and label the original failure test."
        ),
    ),
    # ------------------------------------------------------------------ row 8
    MetricSpec(
        id="constraint_adherence",
        display_name="Constraint Adherence",
        scorecard_row=8,
        tier=Tier.TESTS,
        numerator="Per-problem constraint rules satisfied",
        denominator="Per-problem constraint rules applicable",
        direction=Direction.HIGHER,
        gate_value=0.90,
        blocking=False,
        required_artifacts=(Artifact.RUN_JSON, Artifact.CONSTRAINTS_JSON, Artifact.REFERENCE),
        mlflow_metric_key="constraint_adherence",
        mlflow_scorer_name="constraint_adherence",
        mlflow_feedback_name="constraint_adherence",
        scope=Scope.PER_PROBLEM,
        aggregation=Aggregation.MEAN,
        primary=True,
        notes="Advisory in v1: the 0.90 gate is reported but does not fail the run.",
    ),
    MetricSpec(
        id="complexity_match_rate",
        display_name="Complexity Match Rate (Advisory)",
        scorecard_row=8,
        tier=Tier.TESTS,
        numerator="Problems whose estimated complexity matches or beats the target",
        denominator="Problems with a derivable target and static estimate",
        direction=Direction.HIGHER,
        gate_value=None,
        blocking=False,
        required_artifacts=(Artifact.RUN_JSON, Artifact.REFERENCE, Artifact.CONSTRAINTS_JSON),
        mlflow_metric_key="complexity_match_rate",
        mlflow_scorer_name=None,
        mlflow_feedback_name=None,
        scope=Scope.RUN,
        aggregation=Aggregation.NONE,
        notes=(
            "Advisory only in v1: static loop/recursion estimate, compared with the "
            "parsed O(...) target. Never a hard gate."
        ),
    ),
    # ------------------------------------------------------------------ row 9
    MetricSpec(
        id="infra_error_rate",
        display_name="Infra Error Rate",
        scorecard_row=9,
        tier=Tier.OPS,
        numerator="Runs whose status is infra_error",
        denominator="Runs attempted",
        direction=Direction.LOWER,
        gate_value=0.02,
        blocking=True,
        required_artifacts=(Artifact.RUN_INDEX, Artifact.MANIFEST),
        mlflow_metric_key="infra_error_rate",
        mlflow_scorer_name=None,
        mlflow_feedback_name=None,
        scope=Scope.RUN,
        aggregation=Aggregation.NONE,
        primary=True,
    ),
    MetricSpec(
        id="flaky_rate",
        display_name="Flaky Rate",
        scorecard_row=9,
        tier=Tier.OPS,
        numerator="Repeated problems whose strict verdict changes across repeats",
        denominator="Repeated problems (the 10-problem stratified subset)",
        direction=Direction.LOWER,
        gate_value=0.05,
        blocking=True,
        required_artifacts=(Artifact.RUN_INDEX, Artifact.RUN_JSON, Artifact.MANIFEST),
        mlflow_metric_key="flaky_rate",
        mlflow_scorer_name=None,
        mlflow_feedback_name=None,
        scope=Scope.RUN,
        aggregation=Aggregation.NONE,
    ),
    # ----------------------------------------------------------------- row 10
    MetricSpec(
        id="latency_p95",
        display_name="p95 Latency",
        scorecard_row=10,
        tier=Tier.OPS,
        numerator="95th percentile wall-clock duration per run",
        denominator="Runs with a recorded start and end time",
        direction=Direction.LOWER,
        gate_value=360.0,
        blocking=False,
        required_artifacts=(Artifact.RUN_INDEX, Artifact.TRACE),
        mlflow_metric_key="latency_p95",
        mlflow_scorer_name=None,
        mlflow_feedback_name=None,
        scope=Scope.RUN,
        aggregation=Aggregation.NONE,
        primary=True,
        unit="seconds",
        notes=(
            "Report-only in v1. Repeat-specific keys append _r1/_r2/... so flaky "
            "repeats do not overwrite each other."
        ),
    ),
    MetricSpec(
        id="latency_p50",
        display_name="p50 Latency",
        scorecard_row=10,
        tier=Tier.OPS,
        numerator="Median wall-clock duration per run",
        denominator="Runs with a recorded start and end time",
        direction=Direction.LOWER,
        gate_value=None,
        blocking=False,
        required_artifacts=(Artifact.RUN_INDEX, Artifact.TRACE),
        mlflow_metric_key="latency_p50",
        mlflow_scorer_name=None,
        mlflow_feedback_name=None,
        scope=Scope.RUN,
        aggregation=Aggregation.NONE,
        unit="seconds",
    ),
    MetricSpec(
        id="cost_per_solved",
        display_name="Cost per Solved Problem",
        scorecard_row=10,
        tier=Tier.OPS,
        numerator="Total LLM cost in USD",
        denominator="Strictly solved problems",
        direction=Direction.LOWER,
        gate_value=None,
        blocking=False,
        required_artifacts=(Artifact.TRACE, Artifact.RUN_INDEX),
        mlflow_metric_key="cost_per_solved",
        mlflow_scorer_name=None,
        mlflow_feedback_name=None,
        scope=Scope.RUN,
        aggregation=Aggregation.NONE,
        unit="usd",
        notes=(
            "Trace token_usage/cost first, then evals/pricing.yml fallback. Mark "
            "unavailable rather than logging 0 when neither is present."
        ),
    ),
)


# Public registry. Order follows the scorecard so report tables are stable.
METRICS: dict[str, MetricSpec] = {spec.id: spec for spec in _FIXTURES}

#: The ten IDs that make up the rows of ``scorecard.md``.
CANONICAL_IDS: tuple[str, ...] = tuple(spec.id for spec in _FIXTURES if spec.primary)

#: Primary specs in scorecard order.
PRIMARY_METRICS: tuple[MetricSpec, ...] = tuple(spec for spec in _FIXTURES if spec.primary)

#: MLflow scorer name -> spec (for ``evals.scorers``).
SCORERS: dict[str, MetricSpec] = {
    spec.mlflow_scorer_name: spec
    for spec in _FIXTURES
    if spec.mlflow_scorer_name is not None
}

#: Trace feedback name -> spec (for ``mlflow.log_feedback``).
FEEDBACKS: dict[str, MetricSpec] = {
    spec.mlflow_feedback_name: spec
    for spec in _FIXTURES
    if spec.mlflow_feedback_name is not None
}

#: MLflow metric key -> spec (for ``mlflow.log_metrics``).
METRICS_BY_MLFLOW_KEY: dict[str, MetricSpec] = {
    spec.mlflow_metric_key: spec for spec in _FIXTURES
}


def _validate_registry() -> None:
    """Fail loudly at import time if the contract drifts or duplicates itself."""
    if len(METRICS) != len(_FIXTURES):
        raise ValueError("duplicate metric id in METRICS")
    if len(CANONICAL_IDS) != 10:
        raise ValueError(f"expected 10 primary metrics, found {len(CANONICAL_IDS)}")
    if sorted(spec.scorecard_row for spec in PRIMARY_METRICS) != list(range(1, 11)):
        raise ValueError("each scorecard row must have exactly one primary metric")
    if len(METRICS_BY_MLFLOW_KEY) != len(_FIXTURES):
        raise ValueError("duplicate mlflow_metric_key in METRICS")
    if len(SCORERS) != sum(1 for spec in _FIXTURES if spec.mlflow_scorer_name):
        raise ValueError("duplicate mlflow_scorer_name in METRICS")
    if len(FEEDBACKS) != sum(1 for spec in _FIXTURES if spec.mlflow_feedback_name):
        raise ValueError("duplicate mlflow_feedback_name in METRICS")

    for metric_id, spec in METRICS.items():
        if metric_id != spec.id:
            raise ValueError(f"{metric_id}: registry key does not match spec.id")
        if spec.scope is Scope.PER_PROBLEM and spec.mlflow_scorer_name not in SCORERS:
            raise ValueError(f"{metric_id}: missing scorer registration")
        if spec.scope is Scope.RUN and spec.mlflow_metric_key not in METRICS_BY_MLFLOW_KEY:
            raise ValueError(f"{metric_id}: missing metric registration")


_validate_registry()


def get_metric(metric_id: str) -> MetricSpec:
    """Return one spec, raising a helpful error for an unknown ID."""
    try:
        return METRICS[metric_id]
    except KeyError as exc:
        known = ", ".join(METRICS)
        raise KeyError(f"unknown eval metric {metric_id!r}; known metrics: {known}") from exc


def primary_metrics() -> tuple[MetricSpec, ...]:
    """The ten scorecard rows, in order."""
    return PRIMARY_METRICS


def per_problem_metrics() -> tuple[MetricSpec, ...]:
    """Metrics that become per-row ``mlflow.genai`` scorers and trace feedback."""
    return tuple(spec for spec in _FIXTURES if spec.scope is Scope.PER_PROBLEM)


def run_level_metrics() -> tuple[MetricSpec, ...]:
    """Metrics logged once per eval run with ``mlflow.log_metrics``."""
    return tuple(spec for spec in _FIXTURES if spec.scope is Scope.RUN)


def blocking_metrics() -> tuple[MetricSpec, ...]:
    """Every hard gate, including supporting numbers such as kappa and flaky rate.

    ``report.py`` should evaluate this list rather than only the ten primary rows:
    rows 4 and 9 each carry a second blocking number.
    """
    return tuple(spec for spec in _FIXTURES if spec.blocking)


def scorer_specs() -> dict[str, MetricSpec]:
    """Scorer name -> spec, ready for ``evals.scorers`` decorators."""
    return dict(SCORERS)


def feedback_specs() -> dict[str, MetricSpec]:
    """Feedback name -> spec, ready for ``mlflow.log_feedback``."""
    return dict(FEEDBACKS)


def metric_ids_for_row(row: int) -> tuple[MetricSpec, ...]:
    """Every spec (primary and supporting) belonging to one scorecard row."""
    if row not in range(1, 11):
        raise ValueError("scorecard row must be between 1 and 10")
    return tuple(spec for spec in _FIXTURES if spec.scorecard_row == row)
