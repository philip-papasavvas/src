-- FIRST tutorial on the JOIN operation
-- https://sqlzoo.net/wiki/The_JOIN_operation

-- 1. show the matchid and player name for all goals scored by Germany.
SELECT matchid, player FROM goal
  WHERE teamid ='GER';

-- 3. show the player, teamid, stadium and mdate for every German goal.
SELECT player, teamid, stadium, mdate
  FROM game JOIN goal ON (id=matchid)
  WHERE teamid = 'GER';

-- 4. Show the team1, team2 and player for every goal scored by a player
-- called Mario
SELECT team1, team2, player
  FROM game JOIN goal ON (id=matchid)
  WHERE player LIKE 'Mario%';

-- 5. Show player, teamid, coach, gtime for all goals scored in the
-- first 10 minutes gtime<=10
SELECT player, teamid, coach, gtime
  FROM goal
  JOIN eteam  ON (teamid=id)
  WHERE gtime <= 10;
