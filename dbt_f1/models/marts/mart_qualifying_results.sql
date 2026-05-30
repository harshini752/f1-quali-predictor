-- Qualifying results summary with grid positions and practice pace deltas.
-- One row per driver per race, ordered by qualifying position within each event.

with features as (
    select
        CAST(Year AS INT64)                     as year,
        GrandPrix                               as grand_prix,
        Driver                                  as driver,
        Team                                    as team,
        ROUND(CAST(quali_best AS FLOAT64), 3)   as quali_best_seconds,
        ROUND(CAST(FP1_best AS FLOAT64), 3)     as fp1_best_seconds,
        ROUND(CAST(FP2_best AS FLOAT64), 3)     as fp2_best_seconds,
        ROUND(CAST(FP3_best AS FLOAT64), 3)     as fp3_best_seconds,
        FP1_compound                            as fp1_compound,
        FP2_compound                            as fp2_compound,
        FP3_compound                            as fp3_compound,
        ROUND(CAST(AirTemp AS FLOAT64), 1)      as air_temp_c,
        ROUND(CAST(TrackTemp AS FLOAT64), 1)    as track_temp_c,
        ROUND(CAST(Humidity AS FLOAT64), 1)     as humidity_pct,
        CAST(Rainfall AS BOOL)                  as had_rainfall
    from {{ source('f1_raw', 'qualifying_features') }}
    where quali_best is not null
),

pole_times as (
    select
        year,
        grand_prix,
        MIN(quali_best_seconds) as pole_time_seconds
    from features
    group by year, grand_prix
),

ranked as (
    select
        f.*,
        p.pole_time_seconds,
        -- Gap to pole position
        ROUND(f.quali_best_seconds - p.pole_time_seconds, 3)   as gap_to_pole_seconds,
        -- Practice vs qualifying delta (negative = faster in qualifying)
        ROUND(f.fp3_best_seconds - f.quali_best_seconds, 3)    as fp3_to_quali_delta_seconds,
        ROUND(f.fp1_best_seconds - f.quali_best_seconds, 3)    as fp1_to_quali_delta_seconds,
        RANK() OVER (
            PARTITION BY f.year, f.grand_prix
            ORDER BY f.quali_best_seconds ASC
        )                                                       as quali_position
    from features f
    left join pole_times p
        on f.year = p.year and f.grand_prix = p.grand_prix
)

select
    year,
    grand_prix,
    quali_position,
    driver,
    team,
    quali_best_seconds,
    pole_time_seconds,
    gap_to_pole_seconds,
    fp1_best_seconds,
    fp2_best_seconds,
    fp3_best_seconds,
    fp3_to_quali_delta_seconds,
    fp1_to_quali_delta_seconds,
    fp1_compound,
    fp2_compound,
    fp3_compound,
    air_temp_c,
    track_temp_c,
    humidity_pct,
    had_rainfall
from ranked
order by year, grand_prix, quali_position
