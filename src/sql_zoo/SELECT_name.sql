-- the TABLE world looks like
-- name | region | area | population | gdp
-- see the notes here: https://sqlzoo.net/wiki/Read_the_notes_about_this_table

-- Answers to exercises from https://sqlzoo.net/wiki/SELECT_names

-- 1. find countries that start with "Y"
SELECT name
    FROM world
    WHERE name LIKE 'Y%';

-- 2. find countries that end with "N"
SELECT name
    FROM world
    WHERE name LIKE '%N';

-- 3. find countries that contain the letter "X"
SELECT name
    FROM world
    WHERE name LIKE '%X%';

-- 4. find countries that end with "-LAND'
SELECT name
    FROM world
    WHERE name LIKE '%LAND';

-- 5. find countries that start with a "C" and end in "IA"
SELECT name
    FROM world
    WHERE (name LIKE 'C%')
    AND (name LIKE '%IA');

-- 6. find countries containing double "E" in their names
SELECT name
    FROM world
    WHERE name LIKE '%EE%';

-- 7. find countries containing three A s in their name, e.g. Bahamas
SELECT name
    FROM world
    WHERE name LIKE '%A%A%A';

-- 8. find countries where the second character is "N", order by name
SELECT name
    FROM world
    WHERE name LIKE '_N%'
    ORDER BY name;

-- 9. find countries that have two "O" characters separated by two other characters
SELECT name
    FROM world
    WHERE name LIKE '%OO%';

-- 10. find countries whose names have exactly 4 characters
SELECT name
    FROM world
    WHERE length(name) = 4;

-- 12. find all the countries where the capital is the same name as the country
SELECT name, capital, continent
    FROM world
    WHERE concat(name, ‘ City’) = capital;

-- 13. find the name, capital where the capital includes the name of the country
SELECT name, capital
    FROM world
    WHERE capital LIKE concat('%', name, '%'); -- SQL doesn't like double quotes "

-- 14. find the capital and name where the capital is an extension of the name of the country
SELECT capital, name
    FROM world
    WHERE capital
    LIKE concat(name, '%')
    AND name <> capital; -- filter out where the name of the country is the capital

/* 15. Show the name and extension where the capital is an extension of
the name of the country */
SELECT name, REPLACE(capital, name, '')
    FROM world
    WHERE capital
    LIKE concat('%', name, '%')
    AND name <> capital;