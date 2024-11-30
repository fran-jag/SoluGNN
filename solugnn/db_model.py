from sqlalchemy import REAL, VARCHAR, INTEGER
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class SolvationMolecules(Base):
    __tablename__ = 'train_data'

    mol_id: Mapped[int] = mapped_column(INTEGER(), primary_key=True)
    smiles: Mapped[str] = mapped_column(VARCHAR())
    expt: Mapped[float] = mapped_column(REAL())
