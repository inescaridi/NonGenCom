from nonGenCom.Variables.AgeV1 import AgeV1
from nonGenCom.Variables.AgeV2 import AgeV2

if __name__ == '__main__':
    print("Age V1:")
    age_v1 = AgeV1()  # TODO add test context for age

    likelihood_v1 = age_v1.get_likelihood()
    print("* Likelihood:\n", likelihood_v1)
    likelihood_v1.to_csv("examples/age_example_likelihood_v1.csv")

    posterior = age_v1.get_posterior("Standard")
    print("* Posterior:\n", posterior)
    posterior.to_csv("examples/age_example_posterior.csv")

    # --------------
    print("\nAge V2:")
    age_v2 = AgeV2()

    likelihood_v2 = age_v2.get_likelihood()
    print("* Likelihood v2:\n", likelihood_v2)
    likelihood_v2.to_csv("examples/age_example_likelihood_v2.csv")

    age_v2.set_context('Standard')

    fc_category = 'Adult'
    mp_age = 35
    posterior = age_v2.get_posterior_for_case(fc_category, mp_age)
    print(f"* Posterior for case fc_category {fc_category} and mp_age {mp_age}:\n", posterior)

    ## for testing only
    from_age = 15
    to_age = 24
    mp_age = 25
    posterior = age_v2._get_posterior_for_case_in_range(from_age, to_age, mp_age)
    print(f"* Posterior for case in range {fc_category} and mp_age {mp_age}:\n", posterior)
