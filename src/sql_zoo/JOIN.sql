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

-- 6. List the dates of the matches and the name of
-- the team in which 'Fernando Santos' was the team1 coach.
SELECT game.mdate, eteam.teamname
FROM game
  JOIN eteam ON (game.team1=eteam.id AND eteam.coach LIKE '%Santos');

-- 7. List the player for every goal scored in a game where the
-- stadium was 'National Stadium, Warsaw'
SELECT player FROM game JOIN goal ON (game.id=goal.matchid)
WHERE stadium = 'National Stadium, Warsaw';

-- 8. show the name of all players who scored a goal against Germany
SELECT DISTINCT(player) -- stops players being listed twice
  FROM game JOIN goal ON matchid = id
    WHERE (team1='GER' OR team2='GER')
    AND (teamid !='GER') -- don't show German scorers
