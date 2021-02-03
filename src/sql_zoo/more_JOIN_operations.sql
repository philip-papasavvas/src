-- https://sqlzoo.net/wiki/More_JOIN_operations
-- Database with three tables: movie, actor and casting

-- db table and fields
--
--

 -- 2. When was Citizen Kane released?
 SELECT movie.yr FROM movie
  JOIN actor ON (movie.id=actor.id)
  WHERE movie.title = 'Citizen Kane';

-- 3. List all of the Star Trek movies, include the id, title and yr
-- (all of these movies include the words Star Trek in the title).
-- Order results by year.
SELECT id, title, yr FROM movie
  WHERE title LIKE 'Star Trek%'
  ORDER BY yr;

-- aside: join the casting and actor tables on actorid/id
SELECT * FROM casting JOIN actor
  ON casting.actorid=actor.id;

-- aside: join all of the movie, casting and actor tables
-- first join movie and casting on movie.id=casting.movieid
-- then join on casting.actorid=actor.id
SELECT * FROM
   movie JOIN casting ON movie.id=casting.movieid
         JOIN actor   ON casting.actorid=actor.id;
