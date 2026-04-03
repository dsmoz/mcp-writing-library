"""
User database manager
"""

from datetime import datetime
from typing import List, Optional

import structlog

from kbase.database.pool import DatabasePool
from kbase.models.user import User, UserCreate, UserUpdate

logger = structlog.get_logger(__name__)


class UserManager:
    """
    Manager for user database operations.

    Handles CRUD operations for users and API keys.
    """

    def __init__(self, pool: DatabasePool):
        """
        Initialize user manager.

        Args:
            pool: Database connection pool
        """
        self.pool = pool

    async def create(self, data: UserCreate) -> User:
        """
        Create a new user.

        Args:
            data: User creation data

        Returns:
            Created user

        Raises:
            ValueError: If email already exists
        """
        query = """
            INSERT INTO users (
                id, email, name, is_active, created_at, updated_at, metadata
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            RETURNING *
        """

        now = datetime.utcnow()
        
        # Generate user ID (simple approach - could use UUID or other schemes)
        import uuid
        user_id = f"user-{str(uuid.uuid4())[:8]}"

        try:
            row = await self.pool.fetchrow(
                query,
                user_id,
                data.email,
                data.name,
                True,  # is_active
                now,
                now,
                {},  # metadata
            )

            if not row:
                raise RuntimeError("Failed to create user")

            logger.info(
                "User created",
                user_id=user_id,
                email=data.email,
            )

            return User(**dict(row))

        except Exception as e:
            logger.error("Failed to create user", error=str(e), email=data.email)
            raise

    async def get(self, user_id: str) -> Optional[User]:
        """
        Get a user by ID.

        Args:
            user_id: User ID

        Returns:
            User if found, None otherwise
        """
        query = """
            SELECT * FROM users WHERE id = $1
        """

        try:
            row = await self.pool.fetchrow(query, user_id)

            if not row:
                return None

            return User(**dict(row))

        except Exception as e:
            logger.error("Failed to get user", error=str(e), user_id=user_id)
            raise

    async def get_by_email(self, email: str) -> Optional[User]:
        """
        Get a user by email.

        Args:
            email: User email

        Returns:
            User if found, None otherwise
        """
        query = """
            SELECT * FROM users WHERE email = $1
        """

        try:
            row = await self.pool.fetchrow(query, email)

            if not row:
                return None

            return User(**dict(row))

        except Exception as e:
            logger.error("Failed to get user by email", error=str(e), email=email)
            raise

    async def list(
        self,
        limit: int = 100,
        offset: int = 0,
    ) -> List[User]:
        """
        List all users.

        Args:
            limit: Maximum results
            offset: Result offset for pagination

        Returns:
            List of users
        """
        query = """
            SELECT * FROM users
            ORDER BY created_at DESC
            LIMIT $1 OFFSET $2
        """

        try:
            rows = await self.pool.fetch(query, limit, offset)
            return [User(**dict(row)) for row in rows]

        except Exception as e:
            logger.error("Failed to list users", error=str(e))
            raise

    async def update(
        self,
        user_id: str,
        data: UserUpdate,
    ) -> Optional[User]:
        """
        Update a user.

        Args:
            user_id: User ID
            data: Update data

        Returns:
            Updated user if found, None otherwise
        """
        # Build dynamic update query
        updates = []
        values = []
        param_count = 1

        if data.name is not None:
            updates.append(f"name = ${param_count}")
            values.append(data.name)
            param_count += 1

        if data.is_active is not None:
            updates.append(f"is_active = ${param_count}")
            values.append(data.is_active)
            param_count += 1

        if not updates:
            # No updates provided
            return await self.get(user_id)

        # Add updated_at
        updates.append(f"updated_at = ${param_count}")
        values.append(datetime.utcnow())
        param_count += 1

        # Add user_id as last parameter
        values.append(user_id)

        query = f"""
            UPDATE users
            SET {', '.join(updates)}
            WHERE id = ${param_count}
            RETURNING *
        """

        try:
            row = await self.pool.fetchrow(query, *values)

            if not row:
                return None

            logger.info("User updated", user_id=user_id)
            return User(**dict(row))

        except Exception as e:
            logger.error("Failed to update user", error=str(e), user_id=user_id)
            raise

    async def delete(self, user_id: str) -> bool:
        """
        Delete a user.

        Args:
            user_id: User ID

        Returns:
            True if deleted, False if not found
        """
        query = """
            DELETE FROM users WHERE id = $1
        """

        try:
            result = await self.pool.execute(query, user_id)
            deleted = result == "DELETE 1"

            if deleted:
                logger.info("User deleted", user_id=user_id)

            return deleted

        except Exception as e:
            logger.error("Failed to delete user", error=str(e), user_id=user_id)
            raise

    async def update_last_login(self, user_id: str) -> Optional[User]:
        """
        Update user's last login timestamp.

        Args:
            user_id: User ID

        Returns:
            Updated user if found, None otherwise
        """
        query = """
            UPDATE users
            SET last_login = $1, updated_at = $1
            WHERE id = $2
            RETURNING *
        """

        try:
            now = datetime.utcnow()
            row = await self.pool.fetchrow(query, now, user_id)

            if not row:
                return None

            return User(**dict(row))

        except Exception as e:
            logger.error("Failed to update last login", error=str(e), user_id=user_id)
            raise
