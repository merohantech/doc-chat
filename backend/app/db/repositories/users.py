import uuid

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import User


class UserRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def ensure(self, email: str) -> uuid.UUID:
        """Return the user's id, creating the user if needed. Safe under concurrency."""
        await self._session.execute(
            insert(User).values(id=uuid.uuid4(), email=email).on_conflict_do_nothing()
        )
        user_id = await self._session.scalar(select(User.id).where(User.email == email))
        assert user_id is not None
        return user_id
