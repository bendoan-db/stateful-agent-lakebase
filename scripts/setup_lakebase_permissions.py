"""Grant Lakebase PostgreSQL permissions to a Databricks App's service principal."""

import argparse

from databricks.sdk import WorkspaceClient
from databricks_ai_bridge.lakebase import LakebaseClient

from agent_server.config import APP_NAME, LAKEBASE_INSTANCE_NAME, STORE_TABLES


def grant_permissions(lakebase_instance: str, sp_id: str, workspace_client=None):
    """Execute GRANT statements for the given SP on the Lakebase instance."""
    client = LakebaseClient(
        instance_name=lakebase_instance,
        workspace_client=workspace_client,
    )
    try:
        # Schema-level: USAGE + CREATE (for store.setup() table creation)
        client.execute(f'GRANT USAGE, CREATE ON SCHEMA public TO "{sp_id}"')

        # Table-level: SELECT, INSERT, UPDATE on all store tables
        for table in STORE_TABLES:
            client.execute(
                f'GRANT SELECT, INSERT, UPDATE ON TABLE public.{table} TO "{sp_id}"'
            )

        # Default privileges: auto-grant on future tables created in public schema
        client.execute(
            f"ALTER DEFAULT PRIVILEGES IN SCHEMA public "
            f'GRANT SELECT, INSERT, UPDATE ON TABLES TO "{sp_id}"'
        )
    finally:
        client.close()


def main():
    parser = argparse.ArgumentParser(
        description="Grant Lakebase permissions to a Databricks App's service principal"
    )
    parser.add_argument("--app-name", default=APP_NAME)
    parser.add_argument("--lakebase-instance", default=LAKEBASE_INSTANCE_NAME)
    parser.add_argument(
        "--sp-id", help="Explicit SP client ID (auto-detected from app if omitted)"
    )
    args = parser.parse_args()

    w = WorkspaceClient()

    # Auto-detect SP from app
    if not args.sp_id:
        app = w.apps.get(args.app_name)
        sp_id = app.service_principal_client_id
        print(f"App: {args.app_name}")
        print(f"Service principal: {sp_id}")
    else:
        sp_id = args.sp_id

    print(f"Lakebase instance: {args.lakebase_instance}")
    grant_permissions(args.lakebase_instance, sp_id, workspace_client=w)
    print(f"Permissions granted to {sp_id}")


if __name__ == "__main__":
    main()
