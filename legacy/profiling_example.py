from nonGenCom.BiologicalSex import profiling_biolsex
from nonGenCom.Config import Config

if __name__ == '__main__':
    config = Config()

    context_name = "Female bias"
    scenery_name = "High"

    print(f"Context: {context_name}")
    print(f"Scenery: {scenery_name}")

    prior = config.get_context(context_name)
    likelihood = config.get_scenery(scenery_name)

    cos, cow, ins, inw = profiling_biolsex(likelihood, prior)  # uses default values for cos,cow,ins and inw pairs

    print(f"""
    CoS (strong concistenty) = {cos}
    CoW (weak concistenty) = {cow}
    InS (strong inconcistenty) = {ins}
    InW (weak inconcistenty) = {inw}
    """)


