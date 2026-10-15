"""Export staging that inherits the user's chosen output-folder permissions."""
import uuid


def create_export_stage(parent, prefix='.flp-connect-'):
    # Python 3.13 on Windows applies an owner-only ACL to mkdtemp's mode0700.
    # Rename preserves that ACL, preventing FL Studio under another account
    # from reading a published export. Normal mkdir inherits the parent ACL.
    for _ in range(10):
        stage = parent / (prefix + uuid.uuid4().hex)
        try:
            stage.mkdir()
        except FileExistsError:
            continue
        return stage
    raise FileExistsError('A unique export staging folder could not be created.')
