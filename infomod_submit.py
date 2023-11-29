import time
from functools import reduce 

from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.sql.types import *
from pyspark.sql.functions import *


spark = (SparkSession
            .builder
            .appName("InfoMod")
            .getOrCreate())

spark.sparkContext.setLogLevel("WARN")

COLUMN_NAMES = [] 
ROW_NUM = 1
DIFF_THRESHOLD = 1
SAMPLE_SIZE = 100

def run_pipeline():
   
    raw_data_path = "s3://sparrow-datalake-stage/test/test_data/Living_Room_False_Negative.csv"
    target = "target"
    pred = "pred"

    # ********************* Extract Training patterns *********************
#     v = load_data(target_column = "target", path = raw_data_path)
#     r, _ = train(pattern_name = "target", save_patterns=True)
#     save_csv(r, "patterns")
    
    # ********************* Extract FP patterns *********************
#     v = load_data(target_column = "fp", path = raw_data_path)
#     r, kl = train(pattern_name = "fp", save_patterns=True)
#     save_csv(r, "fp_kitchen_patterns")

    # ********************* Extract FN patterns *********************
    v = load_data(target_column = "fn", path = raw_data_path)
    r, kl = train(pattern_name = "fn", save_patterns=True)
    save_csv(r, "fn_living_room_patterns") 


def load_data(target_column, path):
    
    df = spark.read.option("header", "true").csv(path)
    
    # target                                                                               
    df = df.withColumnRenamed(target_column, "p")

    global COLUMN_NAMES
    COLUMN_NAMES = df.drop("id", "p", "fp", "fn", "pred", "target").columns

    df.createOrReplaceTempView("v")
    return df

############################################ Utility Functions
def add(a, b):
    return a+b

def string_simple_with_reference(ref:"a.", column):
    return f"{ref}{column},"

def string_insert_defeult_v_patterns(column):         
    return f"""CAST(null as string) as {column},"""

def string_draw_random_sample(column):         
    return f"{column},"

def string_table_join_where(table_left, col_left, table_right, col_right):
    return f" ({table_left}.{col_left} IS NULL OR {table_left}.{col_left} = {table_right}.{col_right}) and"

def string_table_join_group(table_left, col_left):
    return f" {table_left}.{col_left},"

def string_table_join_unequal(table_left, col_left, table_right, col_right):
    return f" {table_left}.{col_left} <=> {table_right}.{col_right} and"

def string_table_join_nullif(table_left, col_left, table_right, col_right):
    return f" NULLIF({table_left}.{col_left}, NULLIF({table_left}.{col_left}, {table_right}.{col_right})) as {col_left},"

def string_table_join_nullif_group(table_left, col_left, table_right, col_right):
    return f" NULLIF({table_left}.{col_left}, NULLIF({table_left}.{col_left}, {table_right}.{col_right})),"

def drop_views():
    spark.sql("drop view if exists v_estimate")
    spark.sql("drop view if exists v_sample")
    spark.sql("drop view if exists v_patterns_measure")
    spark.sql("drop view if exists v_patterns_measure_top")
    spark.sql("drop view if exists new_v_patterns")
    spark.sql("drop view if exists new_v_patterns_multiplier")
    spark.sql("drop view if exists v_patterns_lca")
    spark.sql("drop view if exists v_lca_subsets")
    spark.sql("drop view if exists v_lca_measures")
    spark.sql("drop view if exists v_corrected_lca_measures")

def save_final_patterns(view_name):
    if view_name == "":
        return
    v_patterns = spark.sql("select * from v_patterns")
    path = f"""s3://sparrow-datalake-stage/test/test_data/explanation/data/{view_name}.parquet"""
    (v_patterns.write
            .format("parquet")
            .mode("Overwrite")
            .save(path))

def save_csv(df, name):
    (df
     .coalesce(1)
     .sort(col('id').asc())
     .write
     .format("csv")
     .options(header='True', delimiter=',')
     .mode("overwrite")
     .save(f"s3://sparrow-datalake-stage/test/test_data/explanation/result/{name}.csv"))
    
def checkpoint(df, view_name):
    spark.sparkContext.setJobGroup("Writing Jobs", view_name)
    
    path = f"""s3://sparrow-datalake-stage/test/test_data/explanation/data/{view_name}.parquet"""
    (df.write
            .format("parquet")
            .mode("Overwrite")
            .save(path))
    spark.sql(f"""drop view if exists {view_name}""")
    df = spark.read.load(path)
    df.createOrReplaceTempView(view_name)
    return df 


############################################ Algorithm Functions

def train(pattern_name, save_patterns):
    
    global ROW_NUM
    ROW_NUM = 1 

    initial_pattern = create_initial_pattern()
    v_patterns = checkpoint(initial_pattern, "v_patterns")

    print("\n************ Initial patterns ************")
    spark.sql("select * from v_patterns").show(100, truncate=False)

    kl = create_initial_kl()
    kl = checkpoint(kl, "kl")
                                               
    # the loop to create new patterns
    for i in range(1, 10):
        
        drop_views()
        
        print("\n************ Converge ************")
        v_patterns = converge_multipliers()
                                                                                           
        print("\n************ Generated Pattern: ************")
        pattern = generate_pattern()  
        pattern.show(1, truncate=False)                                                              

        # add the highest gain pattern to the current set of patterns
        new_v_patterns = update_v_patterns_by_corrected_measures(pattern)
        new_v_patterns = checkpoint(new_v_patterns, "new_v_patterns")
        
        v_patterns = checkpoint(new_v_patterns, "v_patterns")

        print("\n************ KL: ************")
        updated_kl = calculate_final_kl()
        updated_kl = checkpoint(updated_kl, "updated_kl")
        kl = checkpoint(updated_kl, "kl")
        kl.sort(col('id').asc()).show(100, truncate=False)


    print("\n************ Final Converge ************")
    v_patterns = converge_multipliers()                                                    

    if save_patterns:
        save_pattern_file = "patterns_" + pattern_name  
        save_final_patterns(save_pattern_file)                                        

    print("\n************ Final patterns ************")
    spark.sql("select * from v_patterns").sort(col('id').asc()).show(100, truncate=False)

    print("\n************ Final KL ************")
    updated_kl = calculate_final_kl()
    updated_kl = checkpoint(updated_kl, "updated_kl")
    kl = checkpoint(updated_kl, "kl")
    kl.sort(col('id').asc()).show(100, truncate=False)

    return v_patterns, kl


def converge_multipliers():
    
    v_patterns = spark.sql("select * from v_patterns")
    i = 0
    while True:
        # print(f"Converge Attempt: {i}")
        i = i+1 

        # fit the data with patterns
        v_estimate = generate_v_estimate()
        v_estimate = checkpoint(v_estimate, "v_estimate")
        
        # Compute the loss of each pattern
        v_patterns_measure = generate_v_patterns_measure() 
        v_patterns_measure = checkpoint(v_patterns_measure, "v_patterns_measure")
        
        # calculate the multiplier of high loss patterns
        v_patterns_multiplier = new_v_patterns_multiplier()
        v_patterns_multiplier = checkpoint(v_patterns_multiplier, "v_patterns_multiplier")
        
        # if all losses are below threshold, break
        if v_patterns_multiplier.count()==0:
            break  
        else:
            # update the multiplier of high loss patterns
            new_v_patterns = update_v_patterns()
            new_v_patterns = checkpoint(new_v_patterns, "new_v_patterns")
            
            v_patterns = checkpoint(new_v_patterns, "v_patterns")
        
    return v_patterns


def generate_pattern():
    # draw a new sample
    v_sample = create_sample()
    v_sample = checkpoint(v_sample, "v_sample")
    
    # LCAs
    v_patterns_lca = create_pattern_lca()
    v_patterns_lca = checkpoint(v_patterns_lca, "v_patterns_lca")
    
    # ancestor of LCAs
    v_lca_subsets = create_subsets(v_patterns_lca)
    v_lca_subsets = checkpoint(v_lca_subsets, "v_lca_subsets")
    
    # aggregate over ancestor of LCAs
    v_lca_measures = create_aggregate_patterns()
    v_lca_measures = checkpoint(v_lca_measures, "v_lca_measures")
    
    # correct stats for each pattern 
    v_corrected_lca_measures = create_corrected_measures()
    v_corrected_lca_measures = checkpoint(v_corrected_lca_measures, "v_corrected_lca_measures")

    pattern = next_pattern()

    return pattern


def create_initial_pattern():
    spark.sql("drop view if exists v_patterns")
    v_patterns = spark.sql(     # this will by default insert the null values
        f"""
            select 
                0 as id,
                {reduce(add, map(string_insert_defeult_v_patterns, COLUMN_NAMES))}
                0 as support,
                CAST(0.0 as float) as observation,
                CAST(0.0 as float) as multiplier,
                CAST(0.0 as float) as gain,
                50000 as kl
        """)
    v_patterns.createOrReplaceTempView('v_patterns')
    return v_patterns

def create_sample():
    global SAMPLE_SIZE
    v_sample = spark.sql(f"""
        select 
            a.id,
            {reduce(add, map(string_simple_with_reference, (["a."]*(len(COLUMN_NAMES))), COLUMN_NAMES)).rstrip(',')}
        from v_estimate a
        order by abs(a.p-a.q)/(1-rand(0))
        limit {SAMPLE_SIZE}
    """)
    spark.sql("drop view if exists v_sample")
    v_sample.createOrReplaceTempView('v_sample')
    return v_sample

def generate_v_estimate():
    v_estimate = spark.sql(
        f"""
            SELECT 
                v.id,
                {reduce(add, map(string_table_join_group, (["v"]*(len(COLUMN_NAMES))), COLUMN_NAMES))}
                v.p,
                pow(2, (sum(multiplier))) / (pow(2, (sum(multiplier)))+ 1) as q
            FROM 
                v, 
                v_patterns 
            WHERE 
                {reduce(add, map(string_table_join_where, (["v_patterns"]*(len(COLUMN_NAMES))), COLUMN_NAMES, (["v"]*(len(COLUMN_NAMES))), COLUMN_NAMES)).rstrip('and')}
            GROUP BY 
                v.id, 
                {reduce(add, map(string_table_join_group, (["v"]*(len(COLUMN_NAMES))), COLUMN_NAMES))}
                v.p
    """)
    v_estimate.createOrReplaceTempView('v_estimate')
    return v_estimate 


def generate_v_patterns_measure():
    """ The loss function of each pattern:
        The distribution differene between true and estmate fraction in the support set of each pattern over entire dataset.  
    """
    v_patterns_measure = (spark.sql(
        f"""
        SELECT 
            summ.id, 
            {reduce(add, map(string_table_join_group, (["summ"]*(len(COLUMN_NAMES))), COLUMN_NAMES))}         

            summ.multiplier,    -- pattern's estimate

            avg(p) as p,        -- true fraction
            avg(q) as q,        -- estimate fraction 
            count(*) as ct,     -- support of pattern

            CASE 
                WHEN avg(p) = avg(q) 
                THEN 0.0 

                WHEN (avg(p) = 0.0 and avg(q) = 1.0) 
                THEN count(*) 

                WHEN avg(p) = 0.0 
                THEN (count(*)) * log(2 , 1.0 /(1.0 - avg(q))) 

                WHEN (avg(p) = 1.0 AND avg(q) = 0.0) 
                THEN (count(*)) 

                WHEN avg(p) = 1.0 
                THEN (count(*)) * log(2 , 1.0 / avg(q)) 

                WHEN avg(q) = 0.0 
                THEN (count(*))*(avg(p))* 9 + (count(*))*(1 - avg(p))* log(2 , (1 - avg(p))) 

                WHEN avg(q) = 1.0 
                THEN (count(*)) * (avg(p)) * log(2 , avg(p)) + (count(*)) * (1 - avg(p)) * 9 

                ELSE (count(*)) * (avg(p)) * log(2 , avg(p) / avg(q)) + (count(*)) * (1 - avg(p)) * log(2 , (1 - avg(p)) / (1 - avg(q))) 
            END AS diff    -- the loss of each pattern

        FROM 
            v_estimate est, 
            v_patterns summ 
        WHERE 
            {reduce(add, map(string_table_join_where, (["summ"]*(len(COLUMN_NAMES))), COLUMN_NAMES, (["est"]*(len(COLUMN_NAMES))), COLUMN_NAMES)).rstrip('and')}
        GROUP BY 
            summ.id,
            {reduce(add, map(string_table_join_group, (["summ"]*(len(COLUMN_NAMES))), COLUMN_NAMES))}
            summ.multiplier
            
        """
    ))
    # v_patterns_measure.sort(col('id').asc()).show(100, truncate=False)
    v_patterns_measure.createOrReplaceTempView('v_patterns_measure')
    return v_patterns_measure


def new_v_patterns_multiplier():
    """ This is the main iterative scaling method to find multiplier. """
    global DIFF_THRESHOLD
    v_patterns_multiplier = (spark.sql(
        f"""
        select 
            a.id,
            {reduce(add, map(string_simple_with_reference, (["a."]*(len(COLUMN_NAMES))), COLUMN_NAMES))}

            b.p,
            b.q,

            CASE 
                WHEN p = q  
                THEN multiplier 

                WHEN p = 0.0 
                THEN multiplier - 9 

                WHEN q = 1.0 
                THEN multiplier - 9 

                WHEN p = 1.0 
                THEN multiplier + 9 

                WHEN q = 0.0 
                THEN multiplier + 9 

                ELSE multiplier + log(2 , p  / q ) + log(2 , (1 - q )/(1 - p )) 
            END as new_multiplier 

        from v_patterns a
        inner join (                    -- pick the pattern with highest loss(diff) to fix its multiplier. 
                select id, p, q 
                from v_patterns_measure 
                where diff > {DIFF_THRESHOLD}
                order by diff desc 
                limit 1
            ) b
            on a.id = b.id
    """))
    # v_patterns_multiplier.sort(col('id').asc()).show(100, truncate=False)
    v_patterns_multiplier.createOrReplaceTempView('v_patterns_multiplier')
    return v_patterns_multiplier


def update_v_patterns():
    new_v_patterns = (spark.sql(
    f"""
    select 
        a.id,
        {reduce(add, map(string_simple_with_reference, (["a."]*(len(COLUMN_NAMES))), COLUMN_NAMES))}

        a.support,

        case 
            when a.id = b.id 
            then b.p
            else a.observation
        end as observation,

        case 
            when a.id = b.id 
            then b.new_multiplier
            else a.multiplier
        end as multiplier,

        a.gain,
        a.kl
    from v_patterns a
    left join v_patterns_multiplier b
    on a.id = b.id
    """))
    new_v_patterns.createOrReplaceTempView('new_v_patterns')
    return new_v_patterns 


def create_corrected_measures():
    """
        For each pattern, calculate its gain over the sample dataset not entire dataset.
    """
    v_corrected_lca_measures = (spark.sql(
        f"""
        select
            {reduce(add, map(string_simple_with_reference, (["pat."]*(len(COLUMN_NAMES))), COLUMN_NAMES))}

            -- this pattern has been created by X number of samples X many times. Then divide by sample and you will have the true support in full dataset.
            ct / count(*) as newct,  
            p / ct  as p,   -- ? 
            q / ct  as q,   -- ?

            -- the same gain function in patterns_measure
            CASE 
                WHEN pat.p / ct  = q / ct 
                THEN 0.0 

                WHEN pat.p / ct  = 0.0 
                THEN (ct / count(*)) * (1 - pat.p) * log(2 , (ct - pat.p)/(ct - q)) 

                WHEN pat.p / ct  = 1.0 
                    THEN (ct / count(*)) * (pat.p / ct ) * log(2 , pat.p / q ) 

                ELSE ((ct / count(*)) * (1 - pat.p / ct ) * log(2 , (ct - pat.p)/(ct - q))) + ((ct / count(*)) * (pat.p / ct ) * log(2 , pat.p / q )) 
            END AS gain,    

            CASE 
                WHEN p  = q  
                THEN 0.0 

                WHEN p  = 0 
                    THEN -9 

                ELSE log(2 , p  / q ) 
            END AS multiplier

        FROM 
            v_sample samp, 
            v_lca_measures pat 
        WHERE 
            {reduce(add, map(string_table_join_where, (["pat"]*(len(COLUMN_NAMES))), COLUMN_NAMES, (["samp"]*(len(COLUMN_NAMES))), COLUMN_NAMES)).rstrip('and')}
        GROUP BY 
            {reduce(add, map(string_simple_with_reference, (["pat."]*(len(COLUMN_NAMES))), COLUMN_NAMES))} 
            pat.ct, 
            pat.p, 
            pat.q
        """))
    v_corrected_lca_measures.createOrReplaceTempView('v_corrected_lca_measures')
    return v_corrected_lca_measures

def update_v_patterns_by_corrected_measures(pattern):
    v_patterns = spark.sql("select * from v_patterns")
    
    new_v_patterns = v_patterns.unionAll(pattern)
    new_v_patterns.createOrReplaceTempView('new_v_patterns')
     
    return new_v_patterns

def create_pattern_lca():
    
    v_patterns_lca = (spark.sql(
        f"""
        SELECT 
            {reduce(add, map(string_table_join_nullif, (["v_estimate"]*(len(COLUMN_NAMES))), COLUMN_NAMES, (["v_sample"]*(len(COLUMN_NAMES))), COLUMN_NAMES))}
            count(*) as oct,    -- true support in full dataset
            cast(sum(p) as float) as sump,      -- true in full dataset 
            sum(q) as sumq     -- estimate in full dataset
            
        FROM 
            v_estimate, 
            v_sample 
        group by 
            {reduce(add, map(string_table_join_nullif_group, (["v_estimate"]*(len(COLUMN_NAMES))), COLUMN_NAMES, (["v_sample"]*(len(COLUMN_NAMES))), COLUMN_NAMES)).rstrip(',')}
        """
        ))
    v_patterns_lca.createOrReplaceTempView('v_patterns_lca')
    return v_patterns_lca

def create_aggregate_patterns():
    v_lca_measures = (spark.sql(
        f"""
        select
            {reduce(add, map(string_simple_with_reference, ([""]*(len(COLUMN_NAMES))), COLUMN_NAMES))} 

            sum(oct) as ct,  
            -- sum by the number of times the pattern has been created by matched samples. 
            -- if you divide by # of times in sample, you get back the true support in full dataset. 
            sum(sump) as p, 
            sum(sumq) as q 
        from 
            v_lca_subsets
        group by 
            {reduce(add, map(string_simple_with_reference, ([""]*(len(COLUMN_NAMES))), COLUMN_NAMES)).rstrip(',')}
        """))
    v_lca_measures.createOrReplaceTempView('v_lca_measures')
    return v_lca_measures

def next_pattern():
    global ROW_NUM
    top_v_corrected_lca_measures = (spark.sql(
        f"""
        select 
            a.id,
            {reduce(add, map(string_simple_with_reference, (["a."]*(len(COLUMN_NAMES))), COLUMN_NAMES))}
            a.support, 
            a.observation, 
            a.multiplier, 
            a.gain, 
            a.kl,
            case when (
                {reduce(add, map(string_table_join_unequal, (["a"]*(len(COLUMN_NAMES))), COLUMN_NAMES, (["b"]*(len(COLUMN_NAMES))), COLUMN_NAMES)).rstrip('and')}
                ) 
                then 1 else 0 
            end as flag
        from (
                select 
                    {ROW_NUM} as id,
                    {reduce(add, map(string_simple_with_reference, ([""]*(len(COLUMN_NAMES))), COLUMN_NAMES))} 

                    newct as support,
                    p as observation,
                    multiplier,
                    gain,
                    10000000000 as kl

                from v_corrected_lca_measures 
                order by gain desc 
                limit 100
            ) a
            left join v_patterns b 
                on (
                    {reduce(add, map(string_table_join_unequal, (["a"]*(len(COLUMN_NAMES))), COLUMN_NAMES, (["b"]*(len(COLUMN_NAMES))), COLUMN_NAMES)).rstrip('and')}
                )
        """
        ).where(col("flag") == 0).select("*").drop("flag").sort(col('gain').desc()).limit(1)
    )
    ROW_NUM = ROW_NUM + 1
    return top_v_corrected_lca_measures

def create_subsets(v_patterns_lca):
    """ Create ancestors. keep the true support, p and q. """
    subsets_list = v_patterns_lca.rdd.flatMap(lambda x: rdd_row_subsets(x))
    v_lca_subsets = spark.createDataFrame(subsets_list, v_patterns_lca.schema)
    v_lca_subsets.createOrReplaceTempView('v_lca_subsets')
    
    return v_lca_subsets

def rdd_row_subsets(rdd_row):
    oct = rdd_row.oct 
    sump = rdd_row.sump
    sumq = rdd_row.sumq
    feature_values = list(rdd_row[0 : len(COLUMN_NAMES)])
    
    subsets = list_subsets(feature_values, 0)   # take only feature columns
    set_subsets = set(tuple(i) for i in subsets) # remove duplicates
    list_set_subsets = [list(i) for i in set_subsets]
    
    for i in list_set_subsets:
        i.append(oct)
        i.append(sump)
        i.append(sumq)
    
    return list_set_subsets


def list_subsets(lst, i):
    if i == len(lst):
        return [list(map(lambda x: None, lst))]
    else:
        ret=[]
        half = list_subsets(lst, i+1)
        for r in half:
            ret.append(r)
            # if not lst[i] == None:
            cr = r[:]
            cr[i]=lst[i]
            ret.append(cr)
        return ret

def create_initial_kl():
    kl = (spark.sql(
        f"""
            SELECT 
            current_timestamp() as id,
            50000 as KL
        """))
    kl.createOrReplaceTempView('kl')
    return kl 

def calculate_final_kl():
    global ROW_NUM
    kl = spark.sql("select * from kl")
    new_kl = (spark.sql(
        f"""
        SELECT 
        current_timestamp() as id,
        SUM(
            CASE WHEN p = q  
                THEN 0.0 

                WHEN p  = 1.0 AND q  = 0.0
                THEN 9 
                
                WHEN p  = 1.0 
                THEN - log(2 , q ) 
                
                WHEN p  = 0.0 AND q  = 1.0 
                THEN 9 

                WHEN p  = 0.0 
                THEN - log(2 , (1 - q )) 

                ELSE 9 
            END
        )  as KL
        FROM 
        v_estimate
        """))
    updated_kl = kl.unionAll(new_kl)
    updated_kl.createOrReplaceTempView('updated_kl')
    return updated_kl

############################################################## 

run_pipeline()

