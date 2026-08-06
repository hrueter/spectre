# Distributed under the MIT License.
# See LICENSE.txt for details.

import click
import importlib

# Load subcommands lazily, i.e., only import the module when the subcommand is
# invoked. This is important so the CLI responds quickly.
class LazyGroup(click.Group):
    def __init__(self, *args, lazy_subcommands=None, **kwargs):
        super().__init__(*args, **kwargs)
        # lazy_subcommands is a map of the form:
        #
        #   {command-name} -> {module-name}.{command-object-name}
        #
        self.lazy_subcommands = lazy_subcommands or {}

    def list_commands(self, ctx):
        base = super().list_commands(ctx)
        lazy = sorted(self.lazy_subcommands.keys())
        return base + lazy

    def get_command(self, ctx, cmd_name):
        if cmd_name in self.lazy_subcommands:
            return self._lazy_load(cmd_name)
        return super().get_command(ctx, cmd_name)

    def _lazy_load(self, cmd_name):
        # lazily loading a command, first get the module name and attribute name
        import_path = self.lazy_subcommands[cmd_name]
        modname, cmd_object_name = import_path.rsplit(".", 1)
        # do the import
        mod = importlib.import_module(__name__+"."+modname)
        # get the Command object from that module
        cmd_object = getattr(mod, cmd_object_name)
        # check the result to make debugging easier
        if not isinstance(cmd_object, click.Command):
            raise ValueError(
                f"Lazy loading of {import_path} failed by returning "
                "a non-command object"
            )
        return cmd_object


@click.group(
    name="bbh",
    cls=LazyGroup,
    # for each command provide the path relative to this module
    lazy_subcommands={
      "eccentricity-control":
        "EccentricityControl.eccentricity_control_command",
      "find-horizon": "FindHorizon.find_horizon_command",
      "generate-id": "InitialData.generate_id_command",
      "postprocess-id": "PostprocessId.postprocess_id_command",
      "start-inspiral": "Inspiral.start_inspiral_command",
      "start-ringdown": "Ringdown.start_ringdown_command",
      "run-cce": "Cce.run_cce_command",
    },
    help="Pipeline for binary black hole simulations.",
)
def bbh_pipeline():
    pass

if __name__ == "__main__":
    bbh_pipeline(help_option_names=["-h", "--help"])
