ALTER TABLE misalignment_lab_repair_reviews
  ADD COLUMN repair_scope text,
  ADD COLUMN first_part_shift_seconds double precision,
  ADD COLUMN change_point_seconds double precision,
  ADD COLUMN change_interval_start_seconds double precision,
  ADD COLUMN change_interval_end_seconds double precision,
  ADD COLUMN transition_confirmed boolean;

UPDATE misalignment_lab_repair_reviews
SET repair_scope = CASE
      WHEN estimator_version = 'piecewise-stable-suffix-v1' THEN 'after_change_point'
      ELSE 'global_offset'
    END,
    transition_confirmed = CASE
      WHEN estimator_version != 'piecewise-stable-suffix-v1' AND judgment = 'plausible' THEN TRUE
      ELSE FALSE
    END;

ALTER TABLE misalignment_lab_repair_reviews
  ALTER COLUMN repair_scope SET NOT NULL,
  ALTER COLUMN transition_confirmed SET NOT NULL,
  ADD CONSTRAINT misalignment_lab_repair_reviews_scope_check
    CHECK (repair_scope IN ('global_offset', 'after_change_point')),
  ADD CONSTRAINT misalignment_lab_repair_reviews_transition_check
    CHECK (
      (
        repair_scope = 'global_offset'
        AND first_part_shift_seconds IS NULL
        AND change_point_seconds IS NULL
        AND change_interval_start_seconds IS NULL
        AND change_interval_end_seconds IS NULL
      )
      OR
      (
        repair_scope = 'after_change_point'
        AND (
          (
            first_part_shift_seconds IS NULL
            AND change_point_seconds IS NULL
            AND change_interval_start_seconds IS NULL
            AND change_interval_end_seconds IS NULL
            AND transition_confirmed = FALSE
          )
          OR
          (
            first_part_shift_seconds BETWEEN -12.0 AND 12.0
            AND change_interval_start_seconds >= 0.0
            AND change_interval_end_seconds > change_interval_start_seconds
            AND (
              (
                transition_confirmed = TRUE
                AND change_point_seconds BETWEEN
                  change_interval_start_seconds AND change_interval_end_seconds
              )
              OR
              (transition_confirmed = FALSE AND change_point_seconds IS NULL)
            )
          )
        )
      )
    );
