import uuid
from typing import Annotated

from fastapi import APIRouter, Form, UploadFile, status

from app.api.deps import CurrentUserDep, FileServiceDep, SettingsDep
from app.api.schemas import FileOut
from app.services.file_service import ConflictResolution, FileTooLargeError

router = APIRouter(prefix="/threads/{thread_id}/files", tags=["files"])


@router.post("", response_model=FileOut, status_code=status.HTTP_201_CREATED)
async def upload_file(
    thread_id: uuid.UUID,
    file: UploadFile,
    user_id: CurrentUserDep,
    files: FileServiceDep,
    settings: SettingsDep,
    on_conflict: Annotated[
        ConflictResolution | None,
        Form(
            description=(
                "Required when a different file with the same name exists: "
                "`replace` uploads a new version, `keep_both` stores it under a new name."
            )
        ),
    ] = None,
) -> FileOut:
    """Upload a file. It is sent to the model with the thread's next message."""
    # Read one byte past the limit, so oversized uploads are rejected without
    # reading the rest of the body into memory.
    data = await file.read(settings.max_upload_bytes + 1)
    if len(data) > settings.max_upload_bytes:
        limit_mb = settings.max_upload_bytes / 1024 / 1024
        raise FileTooLargeError(f"Files are limited to {limit_mb:g} MB.")
    uploaded = await files.upload(user_id, thread_id, file.filename or "file", data, on_conflict)
    return FileOut.model_validate(uploaded)


@router.get("", response_model=list[FileOut])
async def list_files(
    thread_id: uuid.UUID, user_id: CurrentUserDep, files: FileServiceDep
) -> list[FileOut]:
    """All files in the thread, including superseded versions, oldest first."""
    return [FileOut.model_validate(f) for f in await files.list_files(user_id, thread_id)]
