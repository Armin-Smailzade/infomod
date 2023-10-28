import time
from functools import reduce 

from pyspark.sql.types import *
from pyspark.sql import Row
from pyspark.sql.functions import *
from pyspark.sql import SparkSession

spark = (SparkSession
            .builder
            .appName("explanation tables")
            .getOrCreate())

spark.sparkContext.setLogLevel("WARN")

############################################ 
def add(a, b):
    return a+b

def string_simple_with_reference(ref:"a.", column):
    return f"{ref}{column},"

def string_insert_defeult_income_summary(column):         
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

############################################ 

def truncate_views():
    spark.sql("drop view if exists income_estimate")
    spark.sql("drop view if exists income_richsummary")
    spark.sql("drop view if exists income_richsummary_top")
    spark.sql("drop view if exists new_income_summary_multiplier")
    spark.sql("drop view if exists new_income_summary")
    spark.sql("drop view if exists income_subsets")
    
    spark.sql("drop view if exists income_sample")
    spark.sql("drop view if exists income_maxpats")
    spark.sql("drop view if exists income_subsets")
    spark.sql("drop view if exists income_aggpats")
    spark.sql("drop view if exists income_corrected")

def save_final_patterns(view_name):
    if view_name == "":
        return
    income_summary = spark.sql("select * from income_summary")
    path = f"""/explanation/data/{view_name}.parquet"""
    (income_summary.write
            .format("parquet")
            .mode("Overwrite")
            .save(path))

def checkpoint(df, view_name):
    spark.sparkContext.setJobGroup("Writing Jobs", view_name)
    
    path = f"""/explanation/data/{view_name}.parquet"""
    (df.write
            .format("parquet")
            .mode("Overwrite")
            .save(path))
    spark.sql(f"""drop view if exists {view_name}""")
    df = spark.read.load(path)
    df.createOrReplaceTempView(view_name)
    return df 

# current_timestamp()
def create_income_summary():
    spark.sql("drop view if exists income_summary")
    income_summary = spark.sql(     # this will by default insert the null values
        f"""
            select 
                0 as id,
                {reduce(add, map(string_insert_defeult_income_summary, column_names))}
                0 as support,
                CAST(0.0 as float) as observation,
                CAST(0.0 as float) as multiplier,
                CAST(0.0 as float) as gain,
                50000 as kl
        """)
    income_summary.createOrReplaceTempView('income_summary')
    return income_summary

def draw_random_sample():
    global sample_size
    random_sample = spark.sql(f"""
        select 
            a.id,
            {reduce(add, map(string_simple_with_reference, (["a."]*(len(column_names))), column_names)).rstrip(',')}
        from income_estimate a
        order by abs(a.p-a.q)/(1-rand(0))
        limit {sample_size}
    """)
    return random_sample

def create_income_sample():
    income_sample = draw_random_sample()
    spark.sql("drop view if exists income_sample")
    income_sample.createOrReplaceTempView('income_sample')
    return income_sample

def generate_income_estimate():
    income_estimate = spark.sql(
        f"""
            SELECT 
                income.id,
                {reduce(add, map(string_table_join_group, (["income"]*(len(column_names))), column_names))}
                income.p,
                pow(2, (sum(multiplier))) / (pow(2, (sum(multiplier)))+ 1) as q
            FROM 
                income, 
                income_summary 
            WHERE 
                {reduce(add, map(string_table_join_where, (["income_summary"]*(len(column_names))), column_names, (["income"]*(len(column_names))), column_names)).rstrip('and')}
            GROUP BY 
                income.id, 
                {reduce(add, map(string_table_join_group, (["income"]*(len(column_names))), column_names))}
                income.p
    """)
    income_estimate.createOrReplaceTempView('income_estimate')
    return income_estimate 


def generate_income_richsummary():
    """ The loss function of each pattern:
        The distribution differene between true and estmate fraction in the support set of each pattern over entire dataset.  
    """
    income_richsummary = (spark.sql(
        f"""
        SELECT 
            summ.id, 
            {reduce(add, map(string_table_join_group, (["summ"]*(len(column_names))), column_names))}         

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
            income_estimate est, 
            income_summary summ 
        WHERE 
            {reduce(add, map(string_table_join_where, (["summ"]*(len(column_names))), column_names, (["est"]*(len(column_names))), column_names)).rstrip('and')}
        GROUP BY 
            summ.id,
            {reduce(add, map(string_table_join_group, (["summ"]*(len(column_names))), column_names))}
            summ.multiplier
            
        """
    ))
    # income_richsummary.sort(col('id').asc()).show(100, truncate=False)
    income_richsummary.createOrReplaceTempView('income_richsummary')
    return income_richsummary


def new_income_summary_multiplier():
    """ This is the main iterative scaling method to find multiplier. """
    global diff
    income_summary_multiplier = (spark.sql(
        f"""
        select 
            a.id,
            {reduce(add, map(string_simple_with_reference, (["a."]*(len(column_names))), column_names))}

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

        from income_summary a
        inner join (                    -- pick the pattern with highest loss(diff) to fix its multiplier. 
                select id, p, q 
                from income_richsummary 
                where diff > {diff}
                order by diff desc 
                limit 1
            ) b
            on a.id = b.id
    """))
    # income_summary_multiplier.sort(col('id').asc()).show(100, truncate=False)
    income_summary_multiplier.createOrReplaceTempView('income_summary_multiplier')
    return income_summary_multiplier


def update_income_summary():
    new_income_summary = (spark.sql(
    f"""
    select 
        a.id,
        {reduce(add, map(string_simple_with_reference, (["a."]*(len(column_names))), column_names))}

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
    from income_summary a
    left join income_summary_multiplier b
    on a.id = b.id
    """))
    new_income_summary.createOrReplaceTempView('new_income_summary')
    return new_income_summary 


def create_income_maxpats():
    
    income_maxpats = (spark.sql(
        f"""
        SELECT 
            {reduce(add, map(string_table_join_nullif, (["income_estimate"]*(len(column_names))), column_names, (["income_sample"]*(len(column_names))), column_names))}
            count(*) as oct,    -- true support in full dataset
            cast(sum(p) as float) as sump,      -- true in full dataset 
            sum(q) as sumq     -- estimate in full dataset
            
        FROM 
            income_estimate, 
            income_sample 
        group by 
            {reduce(add, map(string_table_join_nullif_group, (["income_estimate"]*(len(column_names))), column_names, (["income_sample"]*(len(column_names))), column_names)).rstrip(',')}
        """
        ))
    income_maxpats.createOrReplaceTempView('income_maxpats')
    return income_maxpats



def rec(lst, i):
    if i == len(lst):
        return [list(map(lambda x: None, lst))]
    else:
        ret=[]
        half = rec(lst, i+1)
        for r in half:
            ret.append(r)
            # if not lst[i] == None:
            cr = r[:]
            cr[i]=lst[i]
            ret.append(cr)
        return ret


def subsets(rdd_row):
    oct = rdd_row.oct 
    sump = rdd_row.sump
    sumq = rdd_row.sumq
    feature_values = list(rdd_row[0 : len(column_names)])
    
    subsets = rec(feature_values, 0)   # take only feature columns
    set_subsets = set(tuple(i) for i in subsets) # remove duplicates
    list_set_subsets = [list(i) for i in set_subsets]
    
    for i in list_set_subsets:
        i.append(oct)
        i.append(sump)
        i.append(sumq)
    
    return list_set_subsets

def create_income_subsets(income_maxpats):
    """ Create ancestors. keep the true support, p and q. """
    subsets_list = income_maxpats.rdd.flatMap(lambda x: subsets(x))
    income_subsets = spark.createDataFrame(subsets_list, income_maxpats.schema)
    income_subsets.createOrReplaceTempView('income_subsets')
    
    return income_subsets

def create_income_aggpats():
    income_aggpats = (spark.sql(
        f"""
        select
            {reduce(add, map(string_simple_with_reference, ([""]*(len(column_names))), column_names))} 

            sum(oct) as ct,  
            -- sum by the number of times the pattern has been created by matched samples. 
            -- if you divide by # of times in sample, you get back the true support in full dataset. 
            sum(sump) as p, 
            sum(sumq) as q 
        from 
            income_subsets
        group by 
            {reduce(add, map(string_simple_with_reference, ([""]*(len(column_names))), column_names)).rstrip(',')}
        """))
    income_aggpats.createOrReplaceTempView('income_aggpats')
    return income_aggpats

def create_income_corrected():
    """
        For each pattern, calculate its gain over the sample dataset not entire dataset.
    """
    income_corrected = (spark.sql(
        f"""
        select
            {reduce(add, map(string_simple_with_reference, (["pat."]*(len(column_names))), column_names))}

            -- this pattern has been created by X number of samples X many times. Then divide by sample and you will have the true support in full dataset.
            ct / count(*) as newct,  
            p / ct  as p,   -- ? 
            q / ct  as q,   -- ?

            -- the same gain function in richsummary
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
            income_sample samp, 
            income_aggpats pat 
        WHERE 
            {reduce(add, map(string_table_join_where, (["pat"]*(len(column_names))), column_names, (["samp"]*(len(column_names))), column_names)).rstrip('and')}
        GROUP BY 
            {reduce(add, map(string_simple_with_reference, (["pat."]*(len(column_names))), column_names))} 
            pat.ct, 
            pat.p, 
            pat.q
        """))
    income_corrected.createOrReplaceTempView('income_corrected')
    return income_corrected


def next_pattern():
    global row_num
    top_income_corrected = (spark.sql(
        f"""
        select 
            a.id,
            {reduce(add, map(string_simple_with_reference, (["a."]*(len(column_names))), column_names))}
            a.support, 
            a.observation, 
            a.multiplier, 
            a.gain, 
            a.kl,
            case when (
                {reduce(add, map(string_table_join_unequal, (["a"]*(len(column_names))), column_names, (["b"]*(len(column_names))), column_names)).rstrip('and')}
                ) 
                then 1 else 0 
            end as flag
        from (
                select 
                    {row_num} as id,
                    {reduce(add, map(string_simple_with_reference, ([""]*(len(column_names))), column_names))} 

                    newct as support,
                    p as observation,
                    multiplier,
                    gain,
                    10000000000 as kl

                from income_corrected 
                order by gain desc 
                limit 100
            ) a
            left join income_summary b 
                on (
                    {reduce(add, map(string_table_join_unequal, (["a"]*(len(column_names))), column_names, (["b"]*(len(column_names))), column_names)).rstrip('and')}
                )
        """
        ).where(col("flag") == 0).select("*").drop("flag").sort(col('gain').desc()).limit(1)
    )
    row_num = row_num + 1
    return top_income_corrected


def update_income_summary_by_corrected(pattern):
    income_summary = spark.sql("select * from income_summary")
    
    new_income_summary = income_summary.unionAll(pattern)
    new_income_summary.createOrReplaceTempView('new_income_summary')
     
    return new_income_summary

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
    global row_num
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
        income_estimate
        """))
    updated_kl = kl.unionAll(new_kl)
    updated_kl.createOrReplaceTempView('updated_kl')
    return updated_kl

def converge_multipliers():
    
    income_summary = spark.sql("select * from income_summary")
    i=0
    while True:
        # print(f"Converge Attempt: {i}")
        i = i+1 

        # fit the data with patterns
        income_estimate = generate_income_estimate()
        income_estimate = checkpoint(income_estimate, "income_estimate")
        
        # Compute the loss of each pattern
        income_richsummary = generate_income_richsummary() 
        income_richsummary = checkpoint(income_richsummary, "income_richsummary")
        
        # calculate the multiplier of high loss patterns
        income_summary_multiplier = new_income_summary_multiplier()
        income_summary_multiplier = checkpoint(income_summary_multiplier, "income_summary_multiplier")
        
        # if all losses are below threshold, break
        if income_summary_multiplier.count()==0:
            break  
        else:
            # update the multiplier of high loss patterns
            new_income_summary = update_income_summary()
            new_income_summary = checkpoint(new_income_summary, "new_income_summary")
            
            income_summary = checkpoint(new_income_summary, "income_summary")
        
    return income_summary

def generate_pattern():
    # draw a new sample
    income_sample = create_income_sample()
    income_sample = checkpoint(income_sample, "income_sample")
    
    # LCAs
    income_maxpats = create_income_maxpats()
    income_maxpats = checkpoint(income_maxpats, "income_maxpats")
    
    # ancestor of LCAs
    income_subsets = create_income_subsets(income_maxpats)
    income_subsets = checkpoint(income_subsets, "income_subsets")
    
    # aggregate over ancestor of LCAs
    income_aggpats = create_income_aggpats()
    income_aggpats = checkpoint(income_aggpats, "income_aggpats")
    
    # correct stats for each pattern 
    income_corrected = create_income_corrected()
    income_corrected = checkpoint(income_corrected, "income_corrected")

    pattern = next_pattern()

    return pattern

def save(df, name):
    (df.sort(col('id').asc())
        .repartition(1)
        .write.option("header", True)
        .format("csv")
        .mode('overwrite')
        .save(f"/explanation/patterns/{name}"))

def train(pattern_name, save_patterns):
    
    global row_num
    row_num = 1 

    new_income_summary = create_income_summary()
    income_summary = checkpoint(new_income_summary, "income_summary")

    print("\n************ Initial Summary ************")
    spark.sql("select * from income_summary").show(100, truncate=False)

    kl = create_initial_kl()
    kl = checkpoint(kl, "kl")
                                               
    # the loop to create new patterns
    for i in range(1, 10):
        
        truncate_views()
        
        print("\n************ Converge ************")
        income_summary = converge_multipliers()
                                                                                           
        print("\n************ Generated Pattern: ************")
        pattern = generate_pattern()  
        pattern.show(1, truncate=False)                                                              

        # add the highest gain pattern to the current set of patterns
        new_income_summary = update_income_summary_by_corrected(pattern)
        new_income_summary = checkpoint(new_income_summary, "new_income_summary")
        
        income_summary = checkpoint(new_income_summary, "income_summary")

        print("\n************ KL: ************")
        updated_kl = calculate_final_kl()
        updated_kl = checkpoint(updated_kl, "updated_kl")
        kl = checkpoint(updated_kl, "kl")
        kl.sort(col('id').asc()).show(100, truncate=False)


    print("\n************ Final Converge ************")
    income_summary = converge_multipliers()                                                    

    if save_patterns:
        save_pattern_file = "patterns_" + pattern_name  
        save_final_patterns(save_pattern_file)                                        

    print("\n************ Final Summary ************")
    spark.sql("select * from income_summary").sort(col('id').asc()).show(100, truncate=False)

    print("\n************ Final KL ************")
    updated_kl = calculate_final_kl()
    updated_kl = checkpoint(updated_kl, "updated_kl")
    kl = checkpoint(updated_kl, "kl")
    kl.sort(col('id').asc()).show(100, truncate=False)

    

    return income_summary, kl

############################################################## 

##############################################################

column_names = [] 
row_num = 1
diff = 10
sample_size = 16


def create_income(target_column, path):
    df = (spark
        .read
        .format("csv")
        .options(header=True, inferSchema=True)
        .load(path)
        )
        
    # target                                                                               
    df = df.withColumnRenamed(target_column, "p")
    df = df.where("cat_sex ==  ' Male'")

    global column_names
    column_names = df.drop("id", "p", "fp", "fn", "pred", "target").columns

    df.createOrReplaceTempView("income")
    return df

def read_patterns(target):
    pattern_from_file = "patterns_" + target
    patterns = spark.read.load(f"/explanation/data/{pattern_from_file}.parquet")
    patterns.createOrReplaceTempView('patterns')

    return patterns

def infer_patterns(target, path):

    # read test dataset 
    df = (spark
        .read
        .format("csv")
        .options(header=True, inferSchema=True)
        .load(path)
        )
    df.createOrReplaceTempView("infer")

    # load patterns
    patterns = read_patterns(target)

    global column_names
    column_names = patterns.drop("id", "support", "observation", "multiplier", "gain", "kl").columns

    # calculate metrics by applying patterns on the test dataset
    result = spark.sql(f"""
        
            select
                p.id,
                {reduce(add, map(string_table_join_group, (["p"]*(len(column_names))), column_names))}
                p.observation,

                d.target

            from patterns p 
            inner join infer d on 
                {reduce(add, map(string_table_join_where, (["p"]*(len(column_names))), column_names, (["d"]*(len(column_names))), column_names)).rstrip('and')}
        
        """).withColumn("pattern_type", lit(target)).sort(col('id').asc())

    result.show(25)
    return result

def compare_dist(target, path, model):

    # read test dataset 
    df = (spark
        .read
        .format("csv")
        .options(header=True, inferSchema=True)
        .load(path)
        )
    df.createOrReplaceTempView("test")

    # load patterns
    patterns = patterns = read_patterns(target)

    global column_names
    column_names = patterns.drop("id", "support", "observation", "multiplier", "gain", "kl").columns

    # calculate metrics by applying patterns on the test dataset
    result = spark.sql(f"""
        
            select
                p.id,
                {reduce(add, map(string_table_join_group, (["p"]*(len(column_names))), column_names))}
                
                count(d.target) as volume,

                sum(d.target) / count(d.target) as test_estimate,
                p.observation as pattern_estimate,
                
                ((sum(d.target) / count(d.target)) - (p.observation))/(p.observation) as diff_estimate

            from patterns p 
            left join test d on 
                {reduce(add, map(string_table_join_where, (["p"]*(len(column_names))), column_names, (["d"]*(len(column_names))), column_names)).rstrip('and')}

            group by 
                p.id,
                {reduce(add, map(string_table_join_group, (["p"]*(len(column_names))), column_names))}
                p.observation
        
        """).withColumn("model", lit(model)).sort(col('id').asc())
    
    result.show(25)
    return result
 


def test(target, pred, path, model):

    # read test dataset 
    df = (spark
        .read
        .format("csv")
        .options(header=True, inferSchema=True)
        .load(path)
        )
    
    df.createOrReplaceTempView("prediction")

    # load patterns
    patterns = patterns = read_patterns(target)

    global column_names
    column_names = patterns.drop("id", "support", "observation", "multiplier", "gain", "kl").columns

    # calculate metrics by applying patterns on the test dataset
    result = spark.sql(f"""
        
            select
                p.id,
                {reduce(add, map(string_table_join_group, (["p"]*(len(column_names))), column_names))}
                
                count(d.pred) as volume,
                sum(d.pred) / count(d.pred) as model_estimate,
                sum(d.target) / count(d.pred) as test_real,
                p.observation as target_real,
                ( (sum(d.pred) / count(d.pred)) - (sum(d.target) / count(d.pred)) ) / (sum(d.target) / count(d.pred)) as diff_model_test,
                ( (sum(d.target) / count(d.pred)) - p.observation ) / (p.observation) as diff_test_train,

                sum(case when (pred = 0 AND target = pred) then 1 else 0 end) as TN,    
                sum(case when (pred = 1 AND target = pred) then 1 else 0 end) as TP,
                sum(case when (pred = 0 AND target = 1) then 1 else 0 end) as FN,
                sum(case when (pred = 1 AND target = 0) then 1 else 0 end) as FP,


                (
                    (sum(case when (pred = 1 AND target = pred) then 1 else 0 end)) /
                        (sum(case when (pred = 1 AND target = pred) then 1 else 0 end) +
                        sum(case when (pred = 1 AND target = 0) then 1 else 0 end))
                ) as precision,

                (
                    (sum(case when (pred = 1 AND target = pred) then 1 else 0 end)) /
                        (sum(case when (pred = 1 AND target = pred) then 1 else 0 end) +
                        sum(case when (pred = 0 AND target = 1) then 1 else 0 end))
                ) as recall,

                (
                    (sum(case when (pred = 0 AND target = pred) then 1 else 0 end) + sum(case when (pred = 1 AND target = pred) then 1 else 0 end)) /
                        (sum(case when (pred = 0 AND target = pred) then 1 else 0 end) +
                        sum(case when (pred = 1 AND target = pred) then 1 else 0 end) +
                        sum(case when (pred = 0 AND target = 1) then 1 else 0 end) +
                        sum(case when (pred = 1 AND target = 0) then 1 else 0 end))
                ) as accuracy

            from patterns p 
            left join prediction d on 
                {reduce(add, map(string_table_join_where, (["p"]*(len(column_names))), column_names, (["d"]*(len(column_names))), column_names)).rstrip('and')}

            group by 
                p.id,
                {reduce(add, map(string_table_join_group, (["p"]*(len(column_names))), column_names))}
                p.observation
        
        """).withColumn("model", lit(model)).sort(col('id').asc())
 
    result.show(25)
    return result

def save_csv(df, name):
    df.coalesce(1).sort(col('id').asc()).write.format("csv").options(header='True', delimiter=',').mode("Overwrite").save(f"/explanation/result/{name}.csv")
 
def start():
   
    target = "target"
    pred = "pred"

    # ********************* Extract Training patterns *********************
    income = create_income(target_column = target, path="/explanation/income/train.csv")
    r, kl = train(pattern_name = target, save_patterns=True)
    save_csv(r, "patterns")

    # ********************* Compare Train patterns on Test *********************

    # compare_dist(target, path = f"/explanation/income/test.csv", model="test")


    # ********************* Report Model on Test *********************

    r = test(target, pred, path = f"/explanation/income/xgb.csv", model='xgb')
    save_csv(r, "test_report")

    # ********************* Extract FP patterns *********************
    income = create_income(target_column = "fp", path="/explanation/income/xgb.csv")
    r, kl = train(pattern_name = "fp", save_patterns=True)
    save_csv(r, "fp_patterns")

    # ********************* Extract FN patterns *********************
    income = create_income(target_column = "fn", path="/explanation/income/xgb.csv")
    r, kl = train(pattern_name = "fn", save_patterns=True)
    save_csv(r, "fn_patterns") 
    
    # ********************* Inference *********************
    infer_patterns("target", path = f"/explanation/income/infer.csv")
    infer_patterns("fp", path = f"/explanation/income/infer.csv")
    infer_patterns("fn", path = f"/explanation/income/infer.csv")



start()



