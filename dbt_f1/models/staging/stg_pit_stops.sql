-- Each row where PitInTime is set = a lap that ended with a pit stop.
-- The driver exits the pits on the following lap (PitOutTime on that next lap).
-- We compute pit stop duration as the gap between consecutive stint start times.

with pit_in_laps as (
    select
        Driver                       as driver,
        Team                         as team,
        CAST(Year AS INT64)          as year,
        GrandPrix                    as grand_prix,
        CAST(LapNumber AS INT64)     as pit_in_lap,
        CAST(Stint AS INT64)         as stint_before_stop,
        Compound                     as compound_before_stop,
        CAST(TyreLife AS INT64)      as tyre_age_at_stop,
        PitInTime_seconds            as pit_in_time_seconds,
        AirTemp                      as air_temp_c,
        TrackTemp                    as track_temp_c
    from {{ source('f1_raw', 'raw_lap_times') }}
    where PitInTime_seconds is not null
),

pit_out_laps as (
    select
        Driver                       as driver,
        CAST(Year AS INT64)          as year,
        GrandPrix                    as grand_prix,
        CAST(LapNumber AS INT64)     as pit_out_lap,
        CAST(Stint AS INT64)         as stint_after_stop,
        Compound                     as compound_after_stop,
        PitOutTime_seconds           as pit_out_time_seconds
    from {{ source('f1_raw', 'raw_lap_times') }}
    where PitOutTime_seconds is not null
),

joined as (
    select
        i.driver,
        i.team,
        i.year,
        i.grand_prix,
        i.pit_in_lap,
        i.stint_before_stop,
        o.stint_after_stop,
        i.compound_before_stop,
        o.compound_after_stop,
        i.tyre_age_at_stop,
        i.pit_in_time_seconds,
        o.pit_out_time_seconds,
        -- Total stationary time in the pit lane (seconds)
        ROUND(o.pit_out_time_seconds - i.pit_in_time_seconds, 3) as pit_stop_duration_seconds,
        i.air_temp_c,
        i.track_temp_c
    from pit_in_laps i
    left join pit_out_laps o
        on  i.driver     = o.driver
        and i.year       = o.year
        and i.grand_prix = o.grand_prix
        -- The lap out immediately follows the lap in
        and o.pit_out_lap = i.pit_in_lap + 1
)

select * from joined
where pit_stop_duration_seconds > 0
order by year, grand_prix, driver, pit_in_lap
