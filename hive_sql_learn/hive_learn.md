
### hive工作中用到的一些拼接函数
<https://blog.csdn.net/jsbylibo/article/details/82859168?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-2.channel_param&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-2.channel_param>
1. concat(string s1, string s2, string s3)<br>
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
