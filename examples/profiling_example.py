from nonGenCom.Variables.BiologicalSex import BiologicalSex

if __name__ == '__main__':
    context_name = "Female bias"
    scenery_name = "High"

    print(f"Context: {context_name}")
    print(f"Scenery: {scenery_name}")

    biolsex_var = BiologicalSex()

    prior = biolsex_var.get_context(context_name)
    likelihood = biolsex_var.get_fc_scenery(scenery_name)

    cos, cow, ins, inw = biolsex_var.profiling(prior, likelihood)  # uses default values for cos,cow,ins and inw pairs

    print(f"""
    CoS (strong concistenty) = {cos}
    CoW (weak concistenty) = {cow}
    InS (strong inconcistenty) = {ins}
    InW (weak inconcistenty) = {inw}
    """)


