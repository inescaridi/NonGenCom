from nonGenCom.AgeV1 import AgeV1

if __name__ == '__main__':
    age_var = AgeV1()  # TODO add test context for age

    min_age, max_age, category_ranges = age_var.get_category_ranges()

    likelihood_v1 = age_var.get_likelihood_v1(min_age, max_age, category_ranges)
    print("Likelihood v1:\n", likelihood_v1)
    likelihood_v1.to_csv("age_example_likelihood_v1.csv")

    likelihood_v2 = age_var.get_likelihood_v2(min_age, max_age)
    print("Likelihood v2:\n", likelihood_v2)
    likelihood_v2.to_csv("age_example_likelihood_v2.csv")

    posterior = age_var.get_posterior("Standard")
    print("Posterior:\n", posterior)
    posterior.to_csv("age_example_posterior.csv")

