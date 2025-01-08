import click
import mlrun

from config import workflow_configs
MLRUN_IMAGE = "mlrun/mlrun"

@click.command()
@click.option(
    "--project-name",
    required=True,
    help="Specify the project name.",
)
@click.option(
    "--workflow-name",
    required=True,
    help="Specify the workflow name.",
)
def main(project_name: str, workflow_name: str) -> None:
    project = mlrun.get_or_create_project(
        name=project_name,
        parameters={
            "base_image" : MLRUN_IMAGE,
            "requirements_file" : "requirements.txt",
            "force_build" : False,
            "source" : "s3://mlrun/testing123.zip"
        }
    )

    print(f"Running workflow {workflow_name}...")
    arguments = workflow_configs.get(workflow_name).dict()
    project.run(name=workflow_name, arguments=arguments, dirty=True, watch=True)


if __name__ == "__main__":
    main()