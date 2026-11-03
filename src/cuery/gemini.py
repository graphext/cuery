"""Gemini 3.5 tool-result compatibility for Instructor's validation retries."""

from instructor.core.hooks import Hooks


def repair_function_response_ids(contents):
    """Restore missing result IDs only when the preceding call is unambiguous.

    Instructor 1.14.5 preserves model parts and thought signatures on retry but
    omits the function response ID and uses the unsupported tool role. Never
    reconstruct signed model parts.
    """
    pending = []
    for content in contents or []:
        parts = getattr(content, "parts", None) or []
        if getattr(content, "role", None) == "tool" and any(
            getattr(part, "function_response", None) is not None for part in parts
        ):
            # Gemini accepts user/model roles, including tool-result messages.
            content.role = "user"
        for part in parts:
            call = getattr(part, "function_call", None)
            if call is not None:
                pending.append(call)
            response = getattr(part, "function_response", None)
            if response is None:
                continue
            matches = [
                call
                for call in pending
                if call.name == response.name and (not response.id or call.id == response.id)
            ]
            if len(matches) != 1 or not matches[0].id:
                raise ValueError(
                    f"Cannot unambiguously match Gemini 3.5 function response {response.name!r} "
                    "to a preceding call ID"
                )
            response.id = matches[0].id
            pending.remove(matches[0])


class Gemini35Hooks(Hooks):
    """Validate retry arguments before ordinary hooks, whose errors are swallowed."""

    def emit_completion_arguments(self, *args, **kwargs):
        repair_function_response_ids(kwargs.get("contents"))
        super().emit_completion_arguments(*args, **kwargs)

    def __add__(self, other):
        # Preserve the guard when Instructor combines client and per-call hooks.
        combined = Gemini35Hooks()
        combined += self
        combined += other
        return combined
