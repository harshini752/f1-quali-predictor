with source as (
    select * from {{ source('f1_raw', 'raw_lap_times') }}
),

cleaned as (
    select
        -- Identity
        Driver                          as driver,
        CAST(DriverNumber AS INT64)     as driver_number,
        Team                            as team,
        CAST(Year AS INT64)             as year,
        GrandPrix                       as grand_prix,
        Session                         as session,

        -- Lap metadata
        CAST(LapNumber AS INT64)        as lap_number,
        CAST(Stint AS INT64)            as stint,
        CAST(TyreLife AS INT64)         as tyre_life,
        Compound                        as compound,
        CAST(FreshTyre AS BOOL)         as is_fresh_tyre,
        CAST(IsPersonalBest AS BOOL)    as is_personal_best,

        -- Timing (all in seconds)
        LapTime_seconds                 as lap_time_seconds,
        Sector1Time_seconds             as sector1_seconds,
        Sector2Time_seconds             as sector2_seconds,
        Sector3Time_seconds             as sector3_seconds,
        PitInTime_seconds               as pit_in_time_seconds,
        PitOutTime_seconds              as pit_out_time_seconds,

        -- Speed traps (km/h)
        SpeedI1                         as speed_trap_i1,
        SpeedI2                         as speed_trap_i2,
        SpeedFL                         as speed_trap_fl,
        SpeedST                         as speed_trap_st,

        -- Weather
        AirTemp                         as air_temp_c,
        TrackTemp                       as track_temp_c,
        Humidity                        as humidity_pct,
        CAST(Rainfall AS BOOL)          as had_rainfall,

        -- Position
        CAST(Position AS INT64)         as track_position

    from source
    where
        CAST(IsAccurate AS BOOL) = true
        and LapTime_seconds is not null
        and LapTime_seconds > 0
)

select * from cleaned
