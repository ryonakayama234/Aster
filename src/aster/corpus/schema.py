from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class AsterRecord:
    """Canonical text record used by Aster's corpus pipeline."""

    id: str
    text: str
    source: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)
