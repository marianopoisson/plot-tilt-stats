from .io_utils import load_data,correct_df,filter_rotation,norm_time,save_figure 

def main():
    df = load_data('../data/compare-params-TM3-C.csv')
    df_a=correct_df(df)
    df_b=filter_rotation(df_a,threshold=4)
    df_c=norm_time(df_b)
    df = df_c.copy()
    return df

if __name__ == '__main__':
    main()
