
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
    MAX(IF(REGEXP_CONTAINS(code, "diag_F32"), 1, 0)) outcome,
FROM `uk-biobank-data.EHR.combined_ehr` AS ehr
WHERE before_assessment = False
GROUP BY 1
)

SELECT
    hf._eid,
    outcome.outcome
FROM `uk-biobank-data.assessment.assessment_centre` hf
JOIN inclusion_criteria ic
    ON hf._eid = ic.eid
JOIN outcome
    ON hf._eid = outcome.eid
