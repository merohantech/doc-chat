-- Runs once, when the data volume is first created.
-- Separate database for the test suite so tests never touch dev data.
CREATE DATABASE chat_test OWNER chat;
