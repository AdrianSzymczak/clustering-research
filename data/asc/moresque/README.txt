==========================================================
   MORESQUE  Dataset
==========================================================

MORESQUE (More Sense-tagged QUEries) is a dataset designed for evaluation of
subtopic information retrieval. For details, please refer to our EMNLP-2010
paper:

Roberto Navigli & Giuseppe Crisafulli. Inducing Word Senses to Improve Web
Search Result Clustering. In Proceedings of the 2010 Conference on
Empirical Methods in Natural Language Processing (EMNLP 2010), MIT Stata
Center, Massachusetts, USA, 9-11 October 2010, pp. 116-126.

CONTENTS

This package contains version 1.0 of MORESQUE which consists of 114 topics,
each with a set of subtopics and a list of 100 top-ranking documents.

The topics were selected from the list of ambiguous Wikipedia entries;
i.e., those with "disambiguation" in the title (see
http://en.wikipedia.org/wiki/Wikipedia:Links_to_%28disambiguation%29_pages)

Because the dataset has been developed as a complement for AMBIENT
(http://credo.fub.it/ambient/), it includes queries of length ranged
between 2 and 4 words and numbered from 45.

The 100 documents associated with each topic were collected from the Yahoo!
search engine as of early 2010, and they were subsequently annotated with
subtopic relevance judgments.

The MORESQUE dataset consists of four files where each row is terminated by
Linefeed (ASCII 10) and fields are separated by Tab (ASCII 9). The four
files are described below:

==================== topics.txt ========================

It contains topic ID and description

ID	description
45	the_block
46	stephen_king
47	soul_food
.........
==========================================================



==================== subTopics.txt ========================

It contains subtopic ID (formed by topic ID and subtopic number) and
description; 

ID	description
45.1	The Block (Sydney), the first Aboriginal land handback
45.2	The Block at Orange, an open-air shopping and entertainment mall located in Southern California
45.3	The Block (Baltimore), an adult-entertainment area

.........
==========================================================



==================== results.txt ========================

It contains result ID (formed by topic ID and search engine rank of
result), URL,  title, and snippet

ID	url	title	snippet
45.1	http://www.blockatorange.com/	The Block at Orange	
45.2	http://en.wikipedia.org/wiki/The_Block_(album)	The Block (album) - Wikipedia, the free encyclopedia	The Block was released on September 2, 2008 and debuted at number one on the ... New Kids on the Block · Hangin' Tough · Merry, Merry Christmas · Step by Step ...
45.3	http://www.blockattahoe.com/	The Block at Tahoe	Hotel designed by and for snowboarders. Photo gallery of rooms, details of special events.
.........
==========================================================



==================== STRel.txt ========================

It contains subtopic ID (formed by topic ID and subtopic number) and result
ID (formed by topic ID and search engine rank of result)

subTopicID	resultID
45.2	45.45
45.2	45.24
45.2	45.5
.........
==========================================================

COPYRIGHT AND LICENSE

This work is licensed under the Creative Commons
Attribution-Noncommercial-Share Alike 3.0 Unported License. To view a
copy of this license, visit
http://creativecommons.org/licenses/by-nc-sa/3.0/

or send a letter to

Creative Commons
171 Second Street, Suite 300
San Francisco, California
94105, USA

MORE INFO

If you have any questions or comments, please contact Roberto Navigli
<navigli@di.uniroma1.it> or Antonio Di Marco <dimarco@di.uniroma1.it>.

CHANGES

1.0 First release
