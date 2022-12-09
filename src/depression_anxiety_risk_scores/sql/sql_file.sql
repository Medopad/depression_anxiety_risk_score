
WITH inclusion_criteria AS
(SELECT 
    eid, 
FROM `uk-biobank-data.EHR.combined_ehr`
WHERE before_assessment = True
and REGEXP_CONTAINS(code,'diag_C') 
GROUP BY 1),


outcome AS 
(SELECT 
    ehr.eid,
    MIN(ehr.date) outcome_date,
    MAX(IF(REGEXP_CONTAINS(code, "diag_F32"), 1, 0)) outcome,
FROM `uk-biobank-data.EHR.combined_ehr` AS ehr
WHERE before_assessment = False
GROUP BY 1
),


max_date AS 
(SELECT 
    ehr.eid,
    MAX(ehr.date) max_date,
FROM `uk-biobank-data.EHR.combined_ehr` AS ehr
WHERE before_assessment = False
GROUP BY 1
)


SELECT
    hf._eid,
    hf._191,
    hf._53,
    hf._40000,
    hf._31,
    hf._2090,
    hf._2100,
    hf._2178,
    hf._2050,
    hf._1920,
    hf._6142,
    hf._6145,
    hf._5674,
    hf._1558,
    hf._1960,
    hf._1930,
    hf._4598,
    hf._4581,
    hf._6138,
    hf._20110,
    hf._2010,
    hf._136,
    hf._1980,
    hf._1950,
    hf._1528,
    hf._50,
    hf._1180,
    hf._2110,
    hf._1990,
    hf._22506,
    hf._4631,
    hf._1249,
    hf._738,
    outcome.outcome,
    outcome.outcome_date,
    max_date.max_date,
FROM `uk-biobank-data.assessment.assessment_centre` hf
JOIN inclusion_criteria ic
    ON hf._eid = ic.eid
JOIN outcome
    ON hf._eid = outcome.eid
JOIN max_date
    ON hf._eid = max_date.eid
WHERE _instance_id = 0 and _31 IS NOT NULL
