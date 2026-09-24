"""Policy adapter that labels student-visited states with a shadow teacher."""

from aster.agent.policy import Policy
from aster.agent.selective import SelectivePolicy
from aster.records.intervention import InterventionTrace
from aster.records.trajectory import Trajectory
from aster.records.transition import Action
from aster.runtime.state import RuntimeState


class InterventionPolicy:
    """Observe a SelectivePolicy without changing which Action reaches the runtime.

    The student SelectivePolicy decides the executed Action exactly as before. The teacher
    is queried on the same pre-action state/history only to produce supervision evidence.
    """

    def __init__(
        self,
        student: SelectivePolicy,
        teacher: Policy,
        *,
        teacher_id: str = "rule-v0",
    ):
        self.student = student
        self.teacher = teacher
        self.teacher_id = teacher_id
        self.interventions: list[InterventionTrace] = []

    @property
    def model_id(self) -> str:
        return self.student.model_id

    def decide(
        self,
        state: RuntimeState,
        trajectory: Trajectory,
        available_actions: tuple[str, ...],
    ) -> Action:
        before = len(self.student.routing_traces)
        executed_action = self.student.decide(state, trajectory, available_actions)
        if len(self.student.routing_traces) != before + 1:
            raise RuntimeError("Student SelectivePolicy must append exactly one RoutingTrace")
        routing = self.student.routing_traces[-1]
        if routing.step != state.step or routing.final_action != executed_action:
            raise RuntimeError("Student routing evidence does not match the current decision")

        teacher_action = self.teacher.decide(state, trajectory, available_actions)
        teacher_target_index = next(
            (
                index
                for index, candidate in enumerate(routing.candidates)
                if candidate == teacher_action
            ),
            None,
        )

        self.interventions.append(
            InterventionTrace(
                step=state.step,
                student_model_id=routing.model_id,
                state=state,
                trajectory=trajectory,
                candidates=routing.candidates,
                scores=routing.scores,
                raw_probabilities=routing.raw_probabilities,
                calibrated_probabilities=routing.calibrated_probabilities,
                selected_index=routing.selected_index,
                route=routing.route,
                executed_action=executed_action,
                teacher_id=self.teacher_id,
                teacher_action=teacher_action,
                teacher_target_index=teacher_target_index,
            )
        )
        return executed_action

    def clear_interventions(self) -> None:
        self.interventions.clear()
