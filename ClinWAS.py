import os
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
import time
import sys
import argparse

def test_all(
    df,
    disc_cols=None,
    cont_cols=None,
    infer_type=False,
    output_dir="ClinWAS_output",
    output_type="separate",
    cont_cont_test_method="spearman",
    cont_cont_min_periods=1,
    cont_cont_drop_duplicates=True,
    cont_disc_small_group_action="exclude",
    cont_disc_small_group_threshold=5,
    cont_disc_small_group_threshold_prop=0.01,
    cont_disc_small_group_threshold_logic="upper",
    cont_disc_lumped_group_threshold_int=5,
    cont_disc_test_method="kruskal",
    disc_disc_test_method="chi2",
    disc_disc_drop_duplicates=True,
    correction_method="fdr_bh",
    correction_alpha=0.05
    ):

    '''
    Given the pandas dataframe to test, the column names for discrete and continuous variables and the output directory,
    perform all ClinWAS tests and write them to the output directory. For detail on optional arguments read the 
    doc for each individual testing function
    
    Params:
    -------
    df: pandas dataframe
        The dataframe in which the testing will be done
    disc_cols: list-like
        A list of column names that will be treated as discrete variables
    cont_cols: list-like
        A list of column names that will be treated as continuous variables
    output_dir: str
        The name of the directory to write the output files
    output_type: str
        The string "separate" or "combined". If separate each test is in its own tsv file (recommended). If combined 
        all 3 tests will be lumped together into one tsv file. Defaults to be separated.
    correction_method: str
        Multiple hypothesis correction method to use. For details see statsmodels.stats.multitest.multipletests. Defaults to
        the Benjamini/Hochberg correction method
    correction_alpha: float
        Multiple Hypothesis correction alpha (family-wise error rate) to be used. Defaults to be 0.05.
    Returns:
    --------
    None
    '''

    check_dir = os.path.isdir(output_dir)
    if not check_dir:
        os.makedirs(output_dir)

    start_time = time.time()
    
    if disc_cols == None and cont_cols == None:
        infer_type = True

    if infer_type:
        print("Automatically Inferring data types...")
        disc_types = {"object","bool","category"}
        cont_types = {"int64","float64"}
        disc_cols = []
        cont_cols = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            if dtype in disc_types:
                disc_cols.append(col)
            elif dtype in cont_types:
                cont_cols.append(col)
            else:
                print("Skipping Unknown data type: {col}")
    elif infer_type == False:
        assert disc_cols != None
        assert cont_cols != None

    print(f"{len(disc_cols)} Discrete Columns detected: {disc_cols}")
    print(f"{len(cont_cols)} Continuous Columns detected: {cont_cols}")
    print(f"Total number of rows: {len(df)}, columns: {len(df.columns)}, entries: {len(df)*len(df.columns)}")
    print()

    cont_cont_start_time = time.time()
    print("Performing continuous to continuous testing...")
    # Do continuous to continuous testing?
    cont_to_cont_test_df = cont_to_cont(
        df,
        cont_cols,
        test_method=cont_cont_test_method,
        drop_duplicates=cont_cont_drop_duplicates,
        min_periods=cont_cont_min_periods,
        correction_method=correction_method,
        correction_alpha=correction_alpha
        )
    #print(f"Continuous to continuous testing done. Writing to {output_dir}/cont_to_cont.tsv")
    print(f"Continuous to continuous testing done")
    print(f"Runtime: {time.time()-cont_cont_start_time}s")
    print()

    disc_disc_start_time = time.time()
    print("Performing Discrete to Discrete testing...")
    # Do discrete to discrete testing
    disc_to_disc_test_df = disc_to_disc(
        df,
        disc_cols,
        test_method=disc_disc_test_method,
        drop_duplicates=disc_disc_drop_duplicates,
        correction_method=correction_method,
        correction_alpha=correction_alpha
        )
    #print(f"Discrete to discrete testing done. Writing to {output_dir}/disc_to_disc.tsv")
    print(f"Discrete to discrete testing done.")
    print(f"Runtime: {time.time()-disc_disc_start_time}s")
    print()
    
    cont_disc_run_time = time.time()
    # Do continuous to discrete testing 
    print("Performing Continuous to discrete testing...")
    cont_to_disc_test_df = cont_to_disc(
        df,
        cont_cols,
        disc_cols,
        small_group_action=cont_disc_small_group_action,
        small_group_threshold=cont_disc_small_group_threshold,
        small_group_threshold_prop=cont_disc_small_group_threshold_prop,
        small_group_threshold_logic=cont_disc_small_group_threshold_logic,
        lumped_group_threshold_int=cont_disc_lumped_group_threshold_int,
        test_method=cont_disc_test_method,
        correction_method=correction_method,
        correction_alpha=correction_alpha
        )
    
    #print(f"Continuous to discrete testing done. Writing to {output_dir}/cont_to_disc.tsv")
    print(f"Continuous to discrete testing done")
    print(f"Runtime: {time.time()-cont_disc_run_time}s")
    print()

    total_runtime = time.time()-start_time
    print(f"All tests done. Total run time: {total_runtime}s")
    print(f"Estimated number of entries processed per-second: {len(df)*len(df.columns) / total_runtime}")
    print("Writing results to files...")

    if output_type == "separate":
        cont_to_cont_test_df.to_csv(f"{output_dir}/cont_to_cont.tsv",sep="\t",index=False)
        disc_to_disc_test_df.to_csv(f"{output_dir}/disc_to_disc.tsv",sep="\t",index=False)
        cont_to_disc_test_df.to_csv(f"{output_dir}/cont_to_disc.tsv",sep="\t",index=False)
    elif output_type == "combined":
        combined_df = combine_test_results(
            cont_to_cont_test_df,
            disc_to_disc_test_df,
            cont_to_disc_test_df,
            correction_method=correction_method,
            correction_alpha=correction_alpha
            )
        combined_df.to_csv(f"{output_dir}/tests_combined.tsv",sep="\t",index=False)

    print("Result Writing Complete")
   
    
def cont_to_disc(
    df,
    cont_cols,
    disc_cols,
    small_group_action="exclude",
    small_group_threshold=5,
    small_group_threshold_prop=0.01,
    small_group_threshold_logic="upper",
    lumped_group_threshold_int=5,
    test_method="kruskal",
    correction_method="fdr_bh",
    correction_alpha=0.05
    ):

    '''
    Given a pandas dataframe and columns names of continuous and discrete columns, perform statistical testing such that 
    columns with interesting relationships are reported
    
    Params:
    -------
    df: pandas dataframe
        The dataframe in which the discrete columns will be derived from
    cont_cols: list-like
        Names of the continuous (numerical) columns in which the relationships will be investigated
    disc_cols: list-like
        Names of the discrete columns in which the relationships will be investigated
    small_group_action: String
        What to do when one of the groups has number of observations less than the specified number.
        Currently the only supported solution is to exclude but other solutions may be implemented in the future. 
    small_group_threshold_int: int
        The minimum number of observations required in order for a group to be considered for statistical comparisons.
    small_group_threshold_prop: float
        The minimum proportion of observations required in a subgroup (with respect to the distribution where the comparison is made, NOT overall proportion)
        in order for it to be considered for statistical comparisons. For example, if Group A has 100 observations and the subgroup A1,A2,A3 has 10,40, 50
        observations respectively and this is set to 0.2. Subgroup A1 will not be included in the statistical test.
    small_group_threshold_logic: string
        When both the small_group_threshold_int and small_group_threshold_prop arguments are present, whether the lower or the upper threshold should be used for inclusion.
        This should be one of "lower" or "upper"
    lumped_group_threshold_int:
        When lumping is done, this is the minimum number of observations required in order for statistical comparisons to be done on the lumped category
    test_method: String
        Name of the statistical test to use. Currently only supports "kruskal".
        Other tests and custom functions will be implemented in the future
    # drop_duplicates: bool
    #     When the statistical test is bidirectional, optionally to drop duplicated associations.
    #     Defaults to False.
        
    Returns:
    --------
    cont_to_disc_test_df: pandas dataframe
        A pandas dataframe containing the name of the two column tested and their related statistics
    '''

    # TODO: Upper limit to the number of unique values allowed per column?

    if small_group_threshold_prop:
        assert 0 <= small_group_threshold_prop and small_group_threshold_prop <= 1

    cont_to_disc_test_df_columns = ["Distribution","ConditionalVariable","test_method","groups_compared","group_means","max_mean_diff","test_pval"]
    cont_to_disc_test_df_rows = []

    group_dict = {}


    for cont_col in cont_cols:
        # print(f"Currently Evaluating: {cont_col}")
        cont_samples = [] # TODO turn this into dictionary

        # Records which groups were included in the test for each continuous variable
        group_dict[cont_col] = {}
        non_empty_observations = len(df[~df[cont_col].isna()])

        for disc_col in disc_cols:
            group_dict[cont_col][disc_col] = {}
            # TODO: Think about heavy tail distributions
            # Or group them into single "Other" category
            
            # Groups dataframe by values in each discrete columns
            grouped_disc_dfs = dict(tuple(df.groupby(disc_col)))

            lumped_cont_samples = []

            # Collect the continuous values for all groups
            for cur_group in grouped_disc_dfs:

                # Gets all continuous variables for this subgroup
                cur_group_sample = grouped_disc_dfs[cur_group][cont_col].to_numpy()

                # Check if this group should be included

                if not np.isnan(cur_group_sample).all(): # Check if group is all empty, auto exclude if it is
                    # Check if the current subgroup is considered as a small group                 
                    group_num = len(cur_group_sample)
                    min_int_num = 0 if small_group_threshold == None else small_group_threshold
                    min_prop_num = 0 if small_group_threshold_prop == None else small_group_threshold_prop*non_empty_observations
                    min_num = 0
                    if small_group_threshold_logic == "upper":
                        min_num = min_int_num if min_int_num > min_prop_num else min_prop_num
                    else:
                        min_num = min_int_num if min_int_num < min_prop_num else min_prop_num
                    
                    is_small_group = group_num < min_num
                    
                    # If the current group is a small group, decide on what to do with it
                    if not is_small_group:
                        # print("Not small group")
                        cont_samples.append(cur_group_sample)
                        group_dict[cont_col][disc_col][cur_group] = cur_group_sample
                    elif small_group_action=="lump":
                        lumped_cont_samples.append(cur_group_sample)
                        
                    elif small_group_action=="exclude":
                        continue

                

            # If we decided to lump smaller groups, add all non-empty observations as an individual group
            if small_group_action == "lump":
                lumped_samples = [item for sublist in lumped_cont_samples for item in sublist if not np.isnan(item)]
                #lumped_samples = [sample for sample in lumped_cont_samples if not np.isnan(sample)]
                # The minimum threshold for the lumped category
                if len(lumped_samples) >= lumped_group_threshold_int:
                    cont_samples.append(lumped_samples)
                    group_dict[cont_col][disc_col]["others"] = lumped_samples

            # We can collect other statistics. For now only the maximum difference between means are collected

            groups_compared = group_dict[cont_col][disc_col]
            group_means = {}

            if len(groups_compared) > 0:
                group_means = {key:np.nanmean(groups_compared[key]) for key in groups_compared}
                max_group_mean_diff = np.max(list(group_means.values())) - np.min(list(group_means.values()))
            
            # Perform Kruskal-Wallis Test to test if any distributions are different
            if len(groups_compared) > 1:
                if test_method == "kruskal":
                    try:
                        _,test_pval = stats.kruskal(*list(groups_compared.values()),nan_policy="omit")
                    except ValueError:
                        print(f"Encounterd value error in kruskal upon spliting \'{cont_col}\' by \'{disc_col}\'. Skipping")
                        test_pval = np.NaN
                        max_group_mean_diff = np.NaN
            else: 
                test_pval = np.NaN
                max_group_mean_diff = np.NaN

            cont_to_disc_test_df_rows.append([cont_col,disc_col,test_method,len(groups_compared),str(group_means),max_group_mean_diff,test_pval])
    
    cont_to_disc_test_df = pd.DataFrame(columns=cont_to_disc_test_df_columns,data=cont_to_disc_test_df_rows)

    # Applying multiple hypothesis testing corrections
    _,corrected_pvals,_,_ = multipletests(cont_to_disc_test_df["test_pval"].to_numpy(),alpha=correction_alpha,method=correction_method)
    cont_to_disc_test_df["test_pval_corrected"] = corrected_pvals

    cont_to_disc_test_df = cont_to_disc_test_df.sort_values(by="test_pval")

    return cont_to_disc_test_df

    
def disc_to_disc(
    df,
    disc_cols,
    test_method="chi2",
    correction_method="fdr_bh",
    correction_alpha=0.05,
    drop_duplicates=True
    ):
    '''
    Given a pandas dataframe and columns names of discrete columns, perform statistical testing such that 
    columns with interesting relationships are reported
    
    Params:
    -------
    df: pandas dataframe
        The dataframe in which the discrete columns will be derived from
    disc_cols: list-like
        Names of the discrete columns in which the relationships will be investigated
    test_method: String
        Name of the statistical test to use. Currently only supports "chi2" for chi-sqaured test.
        Other tests and custom functions will be implemented in the future
    drop_duplicates: bool
        When the statistical test is bidirectional, optionally to drop duplicated associations.
        Defaults to False.
        
    Returns:
    --------
    disc_to_disc_test_df: pandas dataframe
        A pandas dataframe containing the name of the two column tested and their related statistics
    '''

    # Do discrete to discrete testing
    # Default: Every pair, in both direction
    # TODO: small group
    # TODO: Highly correlated / hieraichal columns
    # TODO: Assert disc_col in df.columns

    disc_to_disc_test_df_columns = ["variable_1","variable_2","test_method","dof","test_pval"]
    disc_to_disc_test_df_rows = []
    done_disc_col = set() # Avoid going over the same columns

    for main_disc_col in disc_cols:
        done_disc_col.add(main_disc_col)
        for sub_disc_col in disc_cols:
            if not sub_disc_col in done_disc_col:
                if test_method=="chi2":
                    # Make a contingency table using pandas' built in crosstab function
                    crosstab_df = pd.crosstab(df[main_disc_col],df[sub_disc_col]) # This line definitely needs some tests for errors
                    _,chi2_pval,chi2_dof,_ = stats.chi2_contingency(crosstab_df)
                    
                    # Chi-square is bidirectional so the statistics are the same both ways
                    disc_to_disc_test_df_rows.append([sub_disc_col,main_disc_col,test_method,chi2_dof,chi2_pval])
                    if not drop_duplicates:
                        disc_to_disc_test_df_rows.append([main_disc_col,sub_disc_col,test_method,chi2_dof,chi2_pval])
    
    disc_to_disc_test_df = pd.DataFrame(columns=disc_to_disc_test_df_columns,data=disc_to_disc_test_df_rows)

    disc_to_disc_test_df = disc_to_disc_test_df.sort_values(by="test_pval")

    # Applying multiple hypothesis testing corrections
    _,corrected_pvals,_,_ = multipletests(disc_to_disc_test_df["test_pval"].to_numpy(),alpha=correction_alpha,method=correction_method)
    disc_to_disc_test_df["test_pval_corrected"] = corrected_pvals

    return disc_to_disc_test_df

def cont_to_cont(
    df,
    cont_cols,
    test_method="spearman",
    drop_duplicates=True,
    correction_method="fdr_bh",
    correction_alpha=0.05,
    min_periods=1
    ):

    '''
    Given a pandas dataframe and columns names of continuous columns, perform statistical testing such that 
    columns with interesting relationships are reported
    
    Params:
    -------
    df: pandas dataframe
        The dataframe in which the discrete columns will be derived from
    cont_cols: list-like
        Names of the continuous (numerical) columns in which the relationships will be investigated
    test_method: String
        Name of the statistical test to use. Currently supports "pearson", "kendall", or "spearman".
        Other tests and custom functions will be implemented in the future. Defaults to spearman
    drop_duplicates: bool
        When the statistical test is bidirectional, optionally to drop duplicated associations.
        Defaults to False.
    min_periods: int
        Minimum number of observations required per pair of columns to have a valid result. Currently only available for Pearson and Spearman correlation. Optional
        
    Returns:
    --------
    disc_to_disc_test_df: pandas dataframe
        A pandas dataframe containing the name of the two column tested and their related statistics
    '''

    # TODO: keep only 95% Interval?
    # TODO: small group?
    # TODO: Correlation upperbound

    cont_to_cont_test_columns = ["variable_1","variable_2","test_method","correlation","test_pval"]
    cont_to_cont_test_rows = []
    
    df_cont = df[cont_cols].copy()

    if test_method == "pearson":
        df_cont_corr_df = df_cont.corr(method='pearson',min_periods=min_periods)
        df_cont_pval_df = df_cont.corr(method=lambda x, y: stats.pearsonr(x, y)[1]) - np.eye(*df_cont_corr_df.shape)
    elif test_method == "spearman":
        df_cont_corr_df = df_cont.corr(method='spearman',min_periods=min_periods)
        df_cont_pval_df = df_cont.corr(method=lambda x, y: stats.spearmanr(x, y)[1]) - np.eye(*df_cont_corr_df.shape)
    elif test_method == "kendall":
        df_cont_corr_df = df_cont.corr(method='kendall')
        df_cont_pval_df = df_cont.corr(method=lambda x, y: stats.kendalltau(x, y)[1]) - np.eye(*df_cont_corr_df.shape)
    else:
        raise Exception(f"Continuous columns testing Error: test_method given ({test_method}) is not one of ['pearson','spearman','kendall']")
    
    for i in range(len(df_cont.columns)):
        for j in range(i+1,len(df_cont.columns)):
            var_1 = df_cont.columns[j]
            var_2 = df_cont.columns[i]
            corr_num = df_cont_corr_df.iloc[i,j]
            corr_pval = df_cont_pval_df.iloc[i,j]
            cont_to_cont_test_rows.append([var_1,var_2,test_method,corr_num,corr_pval])
            if drop_duplicates == False:
                cont_to_cont_test_rows.append([var_2,var_1,test_method,corr_num,corr_pval])
    
    cont_to_cont_test_df = pd.DataFrame(columns=cont_to_cont_test_columns,data=cont_to_cont_test_rows)

    # Applying multiple hypothesis testing corrections
    _,corrected_pvals,_,_ = multipletests(cont_to_cont_test_df["test_pval"].to_numpy(),alpha=correction_alpha,method=correction_method)
    cont_to_cont_test_df["test_pval_corrected"] = corrected_pvals

    cont_to_cont_test_df = cont_to_cont_test_df.sort_values(by="test_pval")

    return cont_to_cont_test_df


def combine_test_results(
    cont_cont_df,
    disc_disc_df,
    cont_disc_df,
    correction_method="fdr_bh",
    correction_alpha=0.05
    ):
    '''
    Given the three test output dataframes,
    
    Params:
    -------
    cont_cont_df: pandas dataframe
        The dataframe containing test outputs from continuous variables to continuous variables testing
    disc_disc_df: pandas dataframe
        The dataframe containing test outputs from discrete variables to discrete variables testing
    cont_disc_df: pandas dataframe
        The dataframe containing test outputs from continuous variables to discrete variables testing
        
    Returns:
    --------
    test_combined_df: pandas dataframe
        The combined dataframe containing outputs from all the tests
    '''

    cont_cont_df_copy = cont_cont_df.copy()
    cont_cont_df_copy["test_type"] = "cont_cont"
    disc_disc_df_copy = disc_disc_df.copy()
    disc_disc_df_copy["test_type"] = "disc_disc"
    cont_disc_renamed = cont_disc_df.rename(columns={"Distribution":"variable_1","ConditionalVariable":"variable_2"})
    cont_disc_renamed["test_type"] = "cont_disc"
    test_combined_df = pd.concat([cont_cont_df_copy,disc_disc_df_copy,cont_disc_renamed],ignore_index=True)

    # Discard original corrected p-values and re-correct
    _,corrected_pvals,_,_ = multipletests(test_combined_df["test_pval"].to_numpy(),alpha=correction_alpha,method=correction_method)
    test_combined_df["test_pval_corrected"] = corrected_pvals
    print(corrected_pvals)

    test_combined_df = test_combined_df.sort_values(by="test_pval")
    return test_combined_df