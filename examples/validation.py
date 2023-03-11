import pandas as pd


def get_correspondent_mp_id(fc_id, correspondences):
    if fc_id in correspondences.index:
        return correspondences.loc[fc_id][0]
    else:
        print(f"Correspondence of {fc_id} not found")
        return None


def get_corresp_scores(df, correspondences):
    corresp_scores = {}
    for fc_id in df.ID_FC.unique():
        mp_id = get_correspondent_mp_id(fc_id, correspondences)
        corresp_score = df.loc[(df.ID_FC == fc_id) & (df.ID_MP == mp_id)][score_colname].iloc[0]
        corresp_scores[fc_id] = corresp_score

    return pd.Series(corresp_scores)


def count_values(group, thresholds):
    threshold = thresholds[group.name]
    lt = (group < threshold).sum()
    eq = (group == threshold).sum()
    gt = (group > threshold).sum()
    total = len(group)
    return pd.Series({'less_than': lt, 'equal_to': eq, 'greater_than': gt, 'total': total})


if __name__ == '__main__':
    try:
        scores_df = pd.read_csv('examples/scoreCalculator_example_fc_select_output.csv')
    except FileNotFoundError:
        print("Please run example script fc_scoreCalculator_example.py first!")
        raise

    correspondences = pd.read_csv('examples/resources/correspondences.csv').set_index('ID_FC')

    score_colnames = ['biolsex_score', 'age_v1_score', 'age_v2_score', 'Final Score T1', 'Final Score T2']

    for score_colname in score_colnames:
        variable_name = score_colname.split('_')[0]

        corresp_scores = get_corresp_scores(scores_df, correspondences)

        counts = scores_df.groupby('ID_FC')[score_colname].apply(count_values, corresp_scores)

        proportions = counts.loc[:, ['less_than', 'equal_to', 'greater_than']] / counts.loc[:, 'total']
        proportions = proportions.to_frame().unstack(1)
        proportions.columns = proportions.columns.droplevel()

        scores_df = scores_df.merge(proportions.add_suffix(f"_{variable_name}").reset_index(), on='ID_FC')
        print('a')

    # leave only correspondence rows
    scores_df = scores_df.set_index(['ID_FC', 'ID_MP'], drop=False)
    correspondences = correspondences.reset_index().set_index(['ID_FC', 'ID_MP'])

    merged_df = scores_df.join(correspondences, how='right')
    filtered_df = scores_df.loc[scores_df.index.isin(merged_df.index)]

    filtered_df.to_csv("examples/validation.csv", index=False)

    print(filtered_df)
