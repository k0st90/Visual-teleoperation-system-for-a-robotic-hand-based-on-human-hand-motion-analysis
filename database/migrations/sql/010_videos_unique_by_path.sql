ALTER TABLE original_videos
    DROP CONSTRAINT IF EXISTS original_videos_filename_key,
    ADD CONSTRAINT original_videos_full_path_key UNIQUE (full_path);

ALTER TABLE retargeted_videos
    DROP CONSTRAINT IF EXISTS retargeted_videos_filename_key,
    ADD CONSTRAINT retargeted_videos_full_path_key UNIQUE (full_path);
