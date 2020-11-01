
### hive工作中用到的一些拼接函数
<https://blog.csdn.net/jsbylibo/article/details/82859168?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-2.channel_param&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-2.channel_param>
1. **concat**(string s1, string s2, string s3)<br>
这个函数能够把字符串类型的数据连接起来，连接的某个元素可以是列值。<br>
如 concat( aa, ':', bb) 就相当于把aa列和bb列用冒号连接起来了，aa:bb。<br>

2. **cast** 
用法: cast(value as type)<br>
功能: 将某个列的值显示的转化为某个类型<br>
例子: cast(age as string ) 将int类型的数据转化为了String类型<br>

3. **concat_ws**(seperator, string s1, string s2...)<br>
功能: 制定分隔符将多个字符串连接起来，实现“列转行”<br>
例子: 常常结合group by与collect_set使用<br>

4. **collect_set**
用法:collect_set(list等可迭代对象)<br>
功能: 将list去重和变为集合<br>
例子: 将list变为集合<br>

5. **group by**
用法: 将列group起来

### 1-5 例子
现在需要将table中H1,H2 相同的group by 且将 V1 中value转为`string`并去重并使用 `-` 相连
| H1 | H2 | V1 |
|---| ---| ---|
|a|b|1|
|a|b|1|
|a|b|3|
|c|d|4|
|c|d|5|
|c|d|6|

`select H1, H2, concat_w(',', collect_set(cast(c as string)) from table group by H1, H2;`

| H1 | H2 | V1|
|---|---|---|
| a | b| "1,3"|
|d | f | "4,5,6"|

6. **collect_list**
用法: collect_list(list)<br>
功能: 将list变为list
例子: `select H1, H2, collect_list(cast(c as string)) from table group by H1, H2;`<br>

| H1 | H2 | V1|
|---|---|---|
| a | b| ["1", "1", "3"]|
|d | f | ["4", "5", "6"]|

7. 对集合排序 **sort_array()**
```
select H1, H2, sort_array(collect_list(cast(c as string))) from table group by H1, H2;
```



### 表拼接函数
<https://blog.csdn.net/weixin_43619485/article/details/89637715>
1. **join** xxx **on**<br>
功能: 纵向拼接两个表<br>

2. **full join** xxx **on**<br>
功能: 纵向拼接两个表,并且忽略空值<br>

table1
|H1|H2|H3|
|-|-|-|
|50|20|100|
|60|30|110|
<br>

table2
|H1|H2|H3|
|-|-|-|
|50|21|110|
|51|31|120|

```
SELECT 
    T0.H1 as A,
    T0.H2 as B,
    T1.H2 as C,
    t1.H3 + t0.H3 as total
FROM
    (
        SELECT
            H1,
            H2,
            H3
        FROM
            table1
        WHERE
            H2 > 100
    ) as T1
JOIN
    (
        SELECT
            H1,
            H2,
            H3
        FROM
            table2
        WHERE
            H2 > 90
    ) as T2
on 
    T1.H1 = T2.H2
;
``` 
result table<br>
|A|B|C| total|
|-|-|-|-|
|50|20|21|210|
|51|31|?|120|
|60|?|30|110|

    
但是如果数据不能对齐, 则使用full join 进行矫正<br>
```
SELECT
    *
FROM
    table1
FULL JOIN
    table2
on
    table1.H1 = table2.H2
```
|A|B|C| total|
|-|-|-|-|
|50|20|21|210|
|51|31|Null|120|
|60|Null|30|110|


### 映射
**over()** 映射函数<br>
https://blog.csdn.net/qq_22222499/article/details/92406370
|name|h1|h2|h3|
|-|-|-|-|
|a|50|20|100|
|a|60|30|110|
|b|60|31|50|
作用: 将将数组映射给某一列<br>
```
select
    h1,
    h2,
    sum(h3) over() sum_h3
from
    table
```
|name|h1|h2|h3|sum_h3|
|-|-|-|-|-|
|a|50|20|100|260|
|a|60|30|110|260|
|b|60|31|50|260|

条件映射, over(order by id) id=1时, 只选择一行, id=2选择两行
```
select
    h1,
    h2,
    sum(h3) over(order by [1, 2, 3]) sum_h3
from
    table
```
|name|h1|h2|h3|sum_h3|
|-|-|-|-|-|
|a|50|20|100|100|
|a|60|30|110|210|
|b|60|31|50|260|

条件映射, 更加名称划分over(partition by name = name)
```
select
    h1,
    h2,
    sum(h3) over(order by name) sum_h3
from
    table
```
|name|h1|h2|h3|sum_h3|
|-|-|-|-|-|
|a|50|20|100|210|
|a|60|30|110|210|
|b|60|31|50|50|


判断 xxsx 在某个元祖中<br>
xxxx **IN** (1, 2, '3')
例子:  
```
set dt = 2019-10-14-01;

select
xxx
from
xxx 
where concat_ws('-', year, month, day) = '${hiveconf:dt}'
and appid in (1, 2, 3)
and candidate_name in ('a', 'b', 'c');
```



## hive 函数  
https://blog.csdn.net/u011500419/article/details/108835273?utm_medium=distribute.pc_aggpage_search_result.none-task-blog-2~all~sobaiduend~default-1-108835273.nonecase&utm_term=hive%E5%AE%9E%E7%8E%B0decode%E5%87%BD%E6%95%B0&spm=1000.2123.3001.4430

## hive 函数大全
https://blog.csdn.net/weixin_45131142/article/details/95337754?utm_medium=distribute.pc_relevant.none-task-blog-utm_term-3&spm=1001.2101.3001.4242


**row_number() OVER (PARTITION BY COL1 ORDERBY COL2 desc)**  
表示根据COL1分组，在分组内部根据COL2排序，desc倒序, 而此函数计算的值就表示每组内部排序后的顺序编号（该编号在组内是连续并且唯一的)。**当遇到相同排名的时候，不会生成同样的序号，且中间不会空位**
```
ROW_NUMBER() over(PARTITION BY ticket_id ORDER BY utc_update_time DESC) as rn_fare
```

**rank()**
**dense_rank()**
和row_number功能一样，rank()都是分组内统计排名，但是当出现同样排名的时候，**中间会出现空位**。但是dense rank **不会出现空位**
```
select 
    user_id,
    visit_time,
    visit_date,
    rank() over(partition by user_id order by visit_date desc) as rank --每个用户按照访问时间倒排，通常用于统计用户最近一天的访问记录
from wedw_tmp.tmp_url_info
order by user_id,rank
+----------+------------------------+-------------+-------+--+
| user_id  |       visit_time       | visit_date  | rank  |
+----------+------------------------+-------------+-------+--+
| user1    | 2020-09-12 02:20:02.0  | 2020-09-12  | 1     |
| user1    | 2020-09-12 02:20:02.0  | 2020-09-12  | 1     | --同一天访问了两次，9月11号访问排名第三(不是第二, 因为出现了空位)
| user1    | 2020-09-11 11:20:12.0  | 2020-09-11  | 3     |
| user1    | 2020-09-10 08:19:22.0  | 2020-09-10  | 4     |
| user1    | 2020-08-12 19:20:22.0  | 2020-08-12  | 5     |
| user2    | 2020-05-17 06:20:22.0  | 2020-05-17  | 1     |
| user2    | 2020-05-16 19:03:32.0  | 2020-05-16  | 2     |
| user2    | 2020-05-15 12:34:23.0  | 2020-05-15  | 3     |
| user2    | 2020-05-15 13:34:23.0  | 2020-05-15  | 3     |
| user2    | 2020-05-12 18:24:31.0  | 2020-05-12  | 5     |
```

多条件判断  
**case when then end**
举例  
```
case when xxxx then '条件1'
     when xxxx then '条件2'
     when xxxx then '条件3'
else '无符合条件' end as condition

the case -> condition -> 条件2
```



## 字符串相关
1. 拼接  
`concat('a', 'b', 'c')`  
`concat_ws('-', 'a', 'b', 'c')`
2. 查询  
**instr** 查询子字符串
```
--查询vist_time包含10的记录
select 
 user_id,
 visit_time,
 visit_date,
 visit_cnt
from wedw_tmp.tmp_url_info
where instr(visit_time,'10')>0 -- 查询失败返回0, 成功返回子串index起始位置
```
3. 正则表达式  
regexp_extract  
regexp_replace  
```
--将url中？参数后面的内容全部剔除
  select 
    distinct regexp_replace(visit_url,'\\?(.*)','') as visit_url
  from wedw_tmp.tmp_url_info
```

## 条件判断
1. if statement  
`if (a > b, cond1, cond2) as flag`
2. 空值判断 
返回第一非null的值，如果全部都为NULL就返回NULL **COALESCE(column 1, column 2, 3…)**
```

```

## 分配
1. 组数, 分堆 
ntile(), 根据规定数字分组后再进行每个相同classno的集合尽量地平均分5组。 
```
例如：
一班如果有4个同学，则会把这4个同学分成4组1,2,3,4，共4堆，
如果是5个同学，则分成1,2,3,4,5，共5堆。
如果是6个同学，则分成（1,6），2,3,4,5,共5堆。
如果是6个同学，则分成（1,6），（2，7）,3,4,5,共5堆

select t.classno ,t.sno,t.score
ntile(5) over(partition by  classno order by t.sno) as ntile
from t_score as t order by ntile
```


## 拆解
https://blog.csdn.net/guodong2k/article/details/79459282  
1.explode  
table1
|name|list|
|-|-|
|A-B-C|1,2,3|
```
selct explode(split(list, ',')) as expode_list from table1
```
|name|expode_list|
|-|-|
|A-B-C|1|
||2|
||3|

2. 虚拟拆解, 不破坏原始表  

```
select 
name_v2
,name
,list_v2
from table1
LATERAL VIEW explode(split(name,'-'))table_virual as name_v2
LATERAL VIEW explode(split(list,','))table_virual as list_v2
```
|name_v2|name|list_v2|
|-|-|-|
|A|A-B-C|1|
|B||2|
|C||3|


