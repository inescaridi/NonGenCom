from nonGenCom.BiologicalSex import profiling_biolsex
from nonGenCom.Config import Config

if __name__ == '__main__':
    config = Config()

    likelihood = config.get_scenery("Challenged")
    prior = config.get_context("Female bias")

    cos, cow, ins, inw = profiling_biolsex(likelihood, prior) # uses default values for cos,cow,ins and inw pairs

    print(f"""
    CoS (strong concistenty) = {cos}
    CoW (weak concistenty) = {cow}
    InS (strong inconcistenty) = {ins}
    InW (weak inconcistenty) = {inw}
    """)


