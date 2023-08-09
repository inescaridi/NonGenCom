from nonGenCom.Variables.AgeByCategory import AgeByCategory
from nonGenCom.Variables.AgeContinuous import AgeContinuous

if __name__ == '__main__':
    print("Age V1:")
    age_v1 = AgeByCategory()  # TODO add test context for age

    likelihood_v1 = age_v1.get_fc_likelihood()
    print("* Likelihood:\n", likelihood_v1)
    likelihood_v1.to_csv("examples/age_example_likelihood_v1.csv")

    posterior = age_v1.get_fc_posterior("Standard")
    print("* Posterior:\n", posterior)
    posterior.to_csv("examples/age_example_posterior.csv")

    # --------------
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
