"""Environment name ingredient for sacred experiments.

Note: there are separate `environment_name` and `environment` ingredients to avoid
cyclic dependencies between the logger and the environment ingredients.
This is the new dependency structure:

       ┌──────────────────┐
    ┌▶ │ environment_name │
    │  └──────────────────┘
    │    ▲
    │    │
    │    │
    │  ┌──────────────────┐
    │  │      logger      │
    │  └──────────────────┘
    │    ▲
    │    │
    │    │
    │  ┌──────────────────┐
    └─ │   environment    │
       └──────────────────┘


"""
import sacred

environment_name_ingredient = sacred.Ingredient("environment_name")


@environment_name_ingredient.config
def config():
    gym_id = "seals/CartPole-v0"  # The environment to train on

    locals()  # quieten flake8
