import os
import pandas as pd
import numpy as np
import seaborn as sns
import ast

from scipy import stats
from matplotlib import pyplot as plt
from fpdf import FPDF

class PDF(FPDF):
    def get_const(self):
        const = {
            "font": "Arial",
            "ln": 20,
            "title": {
                "cell_width": 80,
                "width": 30,
                "height": 10,
                "text": "ClinWAS",
                "text_size": 15
            },
            "footer": {
                "offset_y": -15,
                "text_size": 8
            },
            "figpage": {
                "cell_width": 200,
                "cell_height":10
            }
        }
        return const
        
    def header(self):
        const = self.get_const()
        
        # Logo
        #self.image('logo_pb.png', 10, 8, 33)
        # Arial bold 15
        self.set_font(const["font"], 'B', const["title"]["text_size"])
        # Move to the right
        self.cell(const["title"]["cell_width"])
        # Title
        self.cell(
            const["title"]["width"],
            const["title"]["height"],
            const["title"]["text"],
            1,
            0,
            'C'
        )
        # Line break
        self.ln(const["ln"])

    # Page footer
    def footer(self):
        const = self.get_const()
        
        # Position at 1.5 cm from bottom
        self.set_y(const["footer"]["offset_y"])
        # Arial italic 8
        self.set_font(const["font"], 'I', const["footer"]["text_size"])
        # Page number
        self.cell(0, 10, 'Page ' + str(self.page_no()) + '/{nb}', 0, 0, 'C')
        
    def add_fig_page(
        self,
        var1,
        var2,
        test_type,
        fig_path=None,
        pval=None,
        pval_adjusted=None,
        test_method=None,
        correlation=None        
    ):
        print("Adding fig page")
        self.add_page()
        const = self.get_const()
        
        page_title = f"Relation tested: {var1}, {var2}"
                
        # Add page title
        self.set_font(const["font"], size = 12)
        self.cell(
            const["figpage"]["cell_width"],
            h=const["title"]["height"],
            txt=page_title,
            ln=1,
            align='L'
        )
        
        # Add page subtitle
        rel_type = None
        if test_type == "cont_disc":
            rel_type = "Continuous vs. Discrete"
        elif test_type == "disc_disc":
            rel_type = "Discrete vs. Discrete"
        elif test_type == "cont_cont":
            rel_type = "Continuous vs. Continuous"
            
        page_sub_title = f"Relation type: {rel_type}"
        
        self.set_font(const["font"], size = 8)
        self.cell(
            const["figpage"]["cell_width"],
            h=const["figpage"]["cell_height"],
            txt=page_sub_title,
            ln=1,
            align='L'
        )
        
        ## TODO: abstraction
        
        # Add image
        #self.cell(const["figpage"]["cell_width"])
        self.image(fig_path,w=120,h=90)

        # Add other testing information

        if test_method is not None:
            if test_method == "kruskal":
                test_method = "Kruskal-Wallis H Test"
            elif test_method == "chi2":
                test_method = "Chi-square Test of Independence"
            test_method_text = f"Test Method: {test_method}"
            self.cell(const["figpage"]["cell_width"],const["figpage"]["cell_height"],txt=test_method_text,ln=1,align='L')
        if pval is not None:
            pval_text = f"p-value: {pval}"
            self.cell(const["figpage"]["cell_width"],const["figpage"]["cell_height"],txt=pval_text,ln=1,align='L')
        if pval_adjusted is not None:
            pval_adjusted_text = f"p-value (adjusted): {pval_adjusted}"
            self.cell(const["figpage"]["cell_width"],const["figpage"]["cell_height"],txt=pval_adjusted_text,ln=1,align='L')
        if correlation is not None:
            if not np.isnan(correlation):
                correlation_text = f"Correlation coefficient: {correlation}"
                self.cell(const["figpage"]["cell_width"],const["figpage"]["cell_height"],txt=correlation_text,ln=1,align='L')

        print("Done adding fig page")
        



def viz_sig_relations(data_df,relation_df,output_dir="clinwas_visualized",pval=0.05,adjusted=True,top=None):
    check_dir = os.path.isdir(output_dir)
    if not check_dir:
        os.makedirs(output_dir)
    p_col = "test_pval_corrected" if adjusted else "test_pval"
    df_to_plot = relation_df[relation_df[p_col]<pval].reset_index()
    if top:
        df_to_plot = df_to_plot[:top]
    
    pdf = PDF(orientation='P', unit='mm', format='letter')
    pdf.alias_nb_pages()
    
    for idx,row in df_to_plot.iterrows():
        print(f"Visualizing relation {idx+1}/{len(df_to_plot)}: {row['variable_1']}|{row['variable_2']}")
        rel_fig = viz_relation(data_df,row)
        fig_path = f"{output_dir}/{idx}|{row['variable_1']}|{row['variable_2']}.png"
        rel_fig.figure.savefig(fig_path,dpi=200,bbox_inches='tight')
        
        pdf.add_fig_page(
            var1=row["variable_1"],
            var2=row["variable_2"],
            test_type=row["test_type"],
            fig_path=fig_path,
            pval=row["test_pval"],
            pval_adjusted=row["test_pval_corrected"],
            test_method=row["test_method"],
            correlation=row["correlation"],
        )
    
        
    print("All significant relationships visualized")
    pdf.output('ClinWAS_Report.pdf', 'F')
    
    

def viz_relation(data,row):
    ax = None
    rel_type = row["test_type"]
    var1 = row["variable_1"]
    var2 = row["variable_2"]
    if rel_type == "disc_disc":
        crosstab_df = pd.DataFrame(ast.literal_eval(row["group_stats"])).T
        ax = viz_disc_disc_row(var1,var2,crosstab_df)
    elif rel_type == "cont_disc":
        group_stats = ast.literal_eval(row["group_stats"])
        ax = viz_cont_disc_row(data,var1,var2,group_stats)
    elif rel_type == "cont_cont":
        ax = viz_cont_cont_row(data,var1,var2)
        
    return ax

def viz_cont_disc_row(data,var1,var2,group_stats):
    
    sns.set(rc={'figure.figsize':(12,8)})
    sns.set_style(style="white")
    include = group_stats.keys()
    df_to_plot = data[data[var2].isin(include)]
    ax = sns.displot(
        data=df_to_plot,
        x=var1,
        hue=var2,
        kde=True,
        common_norm=True
    )
    
    group_stats_df = pd.DataFrame(group_stats)
    group_stats_df = group_stats_df.round(3)

    table_row_labels = [f"{var1}:{label}" for label in  group_stats_df.index]
    table_row_columns = [f"{var2}:{label}" for label in  group_stats_df.columns]
    
    table_plot = plt.table(cellText=group_stats_df.to_numpy(),
                   rowLabels=table_row_labels,
                   colLabels=table_row_columns,
                   transform=plt.gcf().transFigure,
                  )
    
    return ax

def viz_cont_cont_row(data,var1,var2):
    sns.set(rc={'figure.figsize':(12,8)})
    sns.set_style(style="white")
    graph = sns.jointplot(
        data=data,
        x=var1,
        y=var2,
        kind="reg",
        #scatter_kws={"color": "gray"},
        line_kws={"color": "red"}
    )
    x = data[var1].to_numpy()
    y = data[var2].to_numpy()
    nas = np.logical_or(np.isnan(x), np.isnan(y))
    r, p = stats.spearmanr(x[~nas],y[~nas])
    r = "{:.3f}".format(r)
    p = "{:.3f}".format(p)
    
    #plt.legend([f'r = {r}\np = {p}'],bbox_to_anchor=(1,1),ncol=1,title="Spearman")
    graph.ax_joint.set_xlabel(var1)
    graph.ax_joint.set_ylabel(var2)
    
    return graph

def viz_disc_disc_row(var1,var2,crosstab_df):
    sns.set(rc={'figure.figsize':(12,8)})
    sns.set_style(style="white")
    crosstab_df_prop = crosstab_df.div(crosstab_df.sum(axis=1),axis=0)
    
    ax = crosstab_df_prop.plot.bar(stacked=True,rot=0)
    for c in ax.containers:
        labels = ["{:.1f}%".format(v.get_height()*100) if v.get_height() > 0 else '' for v in c]
        ax.bar_label(c, labels=labels, label_type='center',fontsize=20,color="black")
    ax.set_ylim((0,1))
    ax.set_ylabel(f"Proportions")
    ax.set_xlabel(f"{var2}")
    ax.legend(title=f"{var1}")    
    sns.despine()
    
    crosstab_df_T = crosstab_df.T

    table_plot = plt.table(
        cellText=crosstab_df_T.to_numpy(),
        rowLabels=crosstab_df_T.index,
        colLabels=crosstab_df_T.columns,
        transform=plt.gcf().transFigure,
    )
    
    return ax

