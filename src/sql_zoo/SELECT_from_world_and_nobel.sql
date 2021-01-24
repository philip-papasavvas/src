-- Answers to exercises from https://sqlzoo.net/wiki/SELECT_from_WORLD_Tutorial
-- (at the bottom) answers to  some questions from
-- https://sqlzoo.net/wiki/SELECT_from_Nobel_Tutorial

-- 1. Select the name, continent, and population of all countries
SELECT name, continent, population
FROM world;

-- 2. Select countries with more than 200 million population
SELECT name, gdp/population
FROM  world
WHERE population > 200000000; -- quite messy but this is 200 with 6 zeros after

-- 3. Display GDP per capita for countries with more than 200 million population
SELECT name, gdp/population
FROM world
WHERE population > 200000000;

-- 4. Display name, population (in millions) for countries in the continent 'South America'
SELECT name, population/1000000
FROM world
WHERE continent = 'South America';

-- 5. Display the name, population for France, Germany, Italy
SELECT name, population
FROM world
WHERE name IN ('France', 'Germany', 'Italy');

-- 6. Display countries that have the word 'United' in their name
SELECT name
FROM world
WHERE name LIKE '%united';

-- 7. Show the countries (name, population, area) that are big by area (>3 million sq km)
--      or big by population (> 250 million).
SELECT name, population, area
FROM world
WHERE (area > 3000000)
OR (population > 250000000);

-- 8. Exclusive OR (XOR). Show the countries (name, population, area) that are big by area
-- (>3 million sq km) or big by population (> 250 million), but not both.
SELECT name, population, area
FROM world
WHERE (area > 3000000 AND population < 250000000)
OR (population > 250000000 AND area < 3000000);

-- 9 . Show the name and population (in millions) and GDP (in billions) for countries from the
-- continent 'South America'
SELECT name
 ,ROUND(population/1000000, 2)
 ,ROUND(GDP/1000000000, 2)
FROM world
WHERE continent = 'South America';

/* 10. Show the name and per-capita GDP for countries with GDP of more than 1 trillion (12 zeros)
    round the value to the nearest 1000.
     # note the ROUND(f,p) returns f rounded to p decimal places. these can be negative, round
     to nearest 10 when p is -1, nearest 100 for p = -2, etc */
SELECT name, ROUND(gdp/population, -3)
FROM world
where gdp > 1000000000000;

/* 11. Show the name and capital where the name and capital both have the same number of characters */
SELECT name, capital
FROM world
WHERE length(name) = length(capital);

/* 12. Show the name and capital where the first letters of each match, exclude countries where
        they have the same name for the country and capital */
SELECT name, capital
FROM world
WHERE ()LEFT(name, 1) = LEFT(capital, 1))
AND name <> capital;

/* 13. Find the country that has all vowels and no spaces in it's name
        (Dominican Republic and Equatorial Guinea do not count) */
SELECT name
FROM world
WHERE name LIKE '%A%'
AND name LIKE '%E%'
AND name LIKE '%I%'
AND name LIKE '%O%'
AND name LIKE '%U%'
AND name NOT LIKE '% %';        -- name NOT LIKE '%A%' excludes character "A" from results

-- Some non-trivial answers to SELECT_from_nobel
-- 4. Give the name of the 'Peace' award winners since (and incl.) year 2000
SELECT winner
FROM nobel
WHERE subject = 'Peace'
AND yr >= 2000;

-- 5. Show all details (yr, subject, winner) of the Literature prize winners
-- for 1980 to 1989 inclusive
SELECT yr, subject, winner
FROM nobel
WHERE (subject = 'Literature')
AND (yr >= 1980)
AND (yr <= 1989);

-- 7. Show the winners with first name John
SELECT winner
FROM nobel
WHERE winner LIKE 'John%';

-- 9. Show the year, subject, and name of winners
-- for 1980 excluding Chemistry and Medicine
SELECT yr, subject, winner
FROM nobel
WHERE yr = 1980
AND subject NOT IN ('Chemistry', 'Medicine');

-- 10. how year, subject, and name of people who won a 'Medicine' prize in an early year (before 1910, not including 1910) together with winners of a
-- 'Literature' prize in a later year (after 2004, including 2004)
SELECT yr, subject, winner
FROM nobel
WHERE (subject = 'Medicine' AND yr < 1910)
OR (subject = 'Literature' AND yr >= 2004);

--12. Find all details of the prize won by EUGENE O'NEILL
SELECT yr, subject, winner
FROM nobel
WHERE winner = 'EUGENE O''NEILL';
-- escape the quote

-- 13. List the winners, year and subject where the winner starts with
-- Sir. Show the the most recent first, then by name order
SELECT winner, yr, subject
FROM nobel
WHERE winner LIKE 'Sir%'
ORDER BY yr DESC, winner