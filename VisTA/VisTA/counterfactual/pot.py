import json
from typing import Dict, Any, Tuple

import torch
import torch.nn.functional as F

from .llm_client import LLMClient


def build_system_prompt() -> str:
    return (
        "You are a reasoning planner for remote-sensing change QA. "
        "Given a question, produce a JSON plan with fields: vars(optional) and ops(list).\n"
        "Each op is an object with: op (string), args (object), out (string).\n"
        "Supported ops:\n"
        "- SEG_CHANGE: args={source:'E_feat'|'E_pixel', tau: float}  -> produces a binary mask\n"
        "- AREA: args={mask: <var>} -> returns normalized area in [0,1]\n"
        "- THRESHOLD_ANSWER: args={metric:<var>, threshold: float, pos: string, neg: string} -> returns answer string\n"
        "Constraints: Output the final answer in the variable named 'answer'. Return ONLY valid JSON without extra text."
    )


def build_user_prompt(question: str, hints: Dict[str, Any]) -> str:
    """hints may include dataset priors, candidate labels, thresholds, etc."""
    return json.dumps({"question": question, "hints": hints}, ensure_ascii=False)


class PlanExecutor:
    def __init__(self):
        pass

    @staticmethod
    def seg_change(energy: torch.Tensor, tau: float) -> torch.Tensor:
        return (energy >= tau).float()

    @staticmethod
    def area(mask: torch.Tensor) -> torch.Tensor:
        return mask.mean(dim=(-2, -1), keepdim=True)  # (B,1,1,1)

    def run(self, plan: Dict[str, Any], inputs: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
        """
        inputs expects: {'E_feat': Tensor(B,1,H,W) or None, 'E_pixel': Tensor(B,1,H,W)}
        Returns: (variables, final_answer)
        """
        vars: Dict[str, Any] = {}
        vars.update(plan.get("vars", {}))
        for op in plan.get("ops", []):
            name = op.get("op")
            args = op.get("args", {})
            out = op.get("out")
            if name == "SEG_CHANGE":
                source = args.get("source", "E_feat")
                tau = float(args.get("tau", 0.1))
                E = inputs.get(source) or inputs.get("E_pixel")
                if not isinstance(E, torch.Tensor):
                    raise ValueError("Energy map tensor missing")
                vars[out] = self.seg_change(E, tau)
            elif name == "AREA":
                mvar = args.get("mask")
                m = vars.get(mvar)
                if not isinstance(m, torch.Tensor):
                    raise ValueError(f"Mask var {mvar} missing")
                vars[out] = self.area(m)  # (B,1,1,1)
            elif name == "THRESHOLD_ANSWER":
                v = vars.get(args.get("metric"))
                thr = float(args.get("threshold", 0.001))
                pos = str(args.get("pos", "yes"))
                neg = str(args.get("neg", "no"))
                if isinstance(v, torch.Tensor):
                    ans = torch.where(v.flatten() >= thr, 1, 0).item()
                    final = pos if ans == 1 else neg
                else:
                    final = neg
                vars[out] = final
            else:
                raise ValueError(f"Unknown op: {name}")
        final_answer = str(vars.get("answer", ""))
        return vars, final_answer


def llm_plan(question: str, hints: Dict[str, Any], client: LLMClient) -> Dict[str, Any]:
    system = build_system_prompt()
    user = build_user_prompt(question, hints)
    out = client.chat(system, user)
    return json.loads(out)
