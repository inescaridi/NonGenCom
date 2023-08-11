from nonGenCom.Variables.AgeContinuous import AgeContinuous

if __name__ == '__main__':
    print("\nAge V2:")
    age_v2 = AgeContinuous(context_name='Standard')

    likelihood_v2 = age_v2.get_fc_likelihood()
    print("* Likelihood v2:\n", likelihood_v2)
    likelihood_v2.to_csv("examples/age_example_likelihood_v2.csv")

    fc_min_age = 40
    fc_max_age = 40
    mp_age = 35
    posterior = age_v2.get_posterior_for_case(fc_min_age, fc_max_age, mp_age)
    print(f"* Posterior for case fc range ({fc_min_age}, {fc_max_age}) and mp_age {mp_age}:\n", posterior)
