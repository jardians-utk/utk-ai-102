# Module 01- Intoduction to Text Processing in Python
## Topics Covered:
- String in Python
- The String Data Type in Python
- Processing Text Files in Python
- Processing PDF Files in Python
- Regular Expressions in Python
___

#### **You Can't Build a House on a Messy Foundation**

Imagine you’re a chef. You can't cook a gourmet meal with muddy vegetables and unorganized ingredients. You have to wash, chop, and prep everything first. Real-world text is just like that—it's messy. Think about tweets filled with typos and emojis, or long articles with inconsistent formatting.

This is where basic text processing comes in. It’s the "prep work" that turns messy text into clean, usable data for your AI models. For example:

*   **Making Text Consistent:** Simple Python commands like `.lower()` (to make everything lowercase) or `.strip()` (to remove extra spaces) are crucial. Without them, a computer might think "Data" and "data" are two completely different words, which would confuse your program and lead to bad results.
*   **Breaking Text into Pieces:** We need to break down sentences into individual words or symbols, a process called **tokenization**. At its heart, this is just splitting a string of text. Learning to do this with basic Python tools like `.split()` is the first step toward using more advanced NLP libraries later.

If you skip this cleanup stage, any errors or "noise" in your data will get passed down the line, leading to biased or just plain broken AI models.

#### **Working Smart, Not Hard: Speed Matters**

In NLP, you often work with massive amounts of data—like, analyzing millions of customer reviews or all of Wikipedia. If your code is slow, a task that should take minutes could take days.

Understanding the basics of how Python handles text helps you write fast, efficient code. For example, knowing the right way to combine thousands of words into a single document can be the difference between code that runs instantly and code that crashes your computer. This isn't just about being a tidy coder; it's about building applications that can handle real-world challenges at a large scale.

#### **The Building Blocks for All the Cool Stuff**

Learning basic text processing is like learning algebra before you tackle calculus. You can't understand the advanced concepts without a solid grasp of the fundamentals.

Advanced NLP tasks, like figuring out if a movie review is positive or negative (**sentiment analysis**), are built on these basics. Once you understand how to manipulate strings in Python, moving on to powerful libraries like **NLTK** or **Hugging Face** (which professionals use to build state-of-the-art AI) will feel natural and intuitive.

Plus, in a world driven by AI, it’s important to be an ethical programmer. Understanding the basics allows you to look "under the hood" of your programs. You can check your work to make sure you’re not accidentally introducing bias—for example, by ensuring your code handles different languages and accents fairly.

#### **The Bottom Line**

From analyzing patient records in healthcare to tracking financial news, the ability to process text with Python is a superpower. It’s why data scientists obsess over it. A strong foundation lets you build creative and powerful applications, while taking shortcuts leads to fragile systems that fail when you need them most.

In short, learning basic text processing is your key to unlocking the true potential of NLP. It ensures your data is clean, your code is fast, and your AI is reliable. With this foundation, you’re not just shuffling text around—you’re on your way to teaching machines how to truly understand us.
___



### String Data Type in Python

In Python, almost any text you can think of—a username, a tweet, a line from a song—is handled using a **string**. A string is simply a sequence of characters: letters, numbers, symbols, and spaces, all bundled together.

Think of strings as the fundamental building block for any task involving text. Before you can analyze a tweet's sentiment or have a chatbot understand a question, you need to know how to work with strings.

One key rule to remember: in Python, strings are **immutable**. This sounds complicated, but it just means that once you create a string, you can't change it. Think of it like a printed photograph. You can't erase part of the photo, but you can always take a *new* photo with changes, or put the original in a different frame. This makes your code more predictable and, behind the scenes, more efficient.

#### How to Make a String

You can create strings using single quotes (`'...'`), double quotes (`"..."`), or triple quotes (`"""..."""`) for text that spans multiple lines.

```python
# Single or double quotes work for most things.
# Use the one that's most convenient!
greeting = "Hello, World!"
fact = 'Python is named after Monty Python, not the snake.'

# If your string contains quotes, use the other type to wrap it.
quote = 'She said, "Python is fun!"'

# Use triple quotes for multi-line text, like song lyrics or a poem.
poem = """Roses are red,
Violets are blue,
This is a multi-line string,
It's pretty cool!"""
```

#### Combining and Repeating Strings

You can do some basic math-like operations with strings.

*   **Concatenation (`+`):** Joins strings together.
*   **Repetition (`*`):** Repeats a string multiple times.

```python
first_name = "Ada"
last_name = "Lovelace"

# Use + to join strings (don't forget the space!)
full_name = first_name + " " + last_name
print(full_name)  # Output: Ada Lovelace

# Use * to repeat a string
echo = "Go! " * 3
print(echo)  # Output: Go! Go! Go!
```

**Pro-Tip:** While `+` is fine for joining a couple of strings, it can be very slow if you're building a huge string inside a loop. We'll cover a much faster method called `.join()` in a bit!

#### Grabbing Parts of a String (Indexing and Slicing)

Since a string is a sequence, you can grab individual characters or sections of it. Python starts counting from **0**, not 1.

```python
text = "Python"

# Get the character at index 0 (the first character)
print(text[0])    # Output: 'P'

# Use negative numbers to count from the end. -1 is the last character.
print(text[-1])   # Output: 'n'

# Slicing lets you grab a "slice" of the string: [start:end]
# Note: It goes UP TO, but does not include, the 'end' index.
substring = text[1:4]  # Grabs characters at index 1, 2, and 3
print(substring)       # Output: 'yth'

# A cool trick to reverse a string!
reversed_text = text[::-1]
print(reversed_text)   # Output: 'nohtyP'
```

#### Your String Toolkit: Super-Useful Methods

Python strings come with a ton of built-in functions called **methods** that act like a toolkit for transforming text. You call them using a dot (`.`) after your string variable.

**1. Changing Case**
Great for making text consistent before you analyze it.

```python
message = "this is a test"
print(message.upper())     # Output: "THIS IS A TEST"
print(message.title())     # Output: "This Is A Test"
```

**2. Cleaning Up Whitespace**
Perfect for cleaning up user input from a form.

```python
# .strip() removes spaces, tabs, or newlines from the beginning and end.
dirty_text = "   some data   \n"
clean_text = dirty_text.strip()
print(clean_text)          # Output: "some data"
```

**3. Splitting and Joining**
This is one of the most powerful combos for processing text. `.split()` breaks a string into a list of smaller strings, and `.join()` does the opposite.

```python
sentence = 'Text processing is fun'
words = sentence.split(' ')  # Split the string wherever there's a space
print(words)                 # Output: ['Text', 'processing', 'is', 'fun']

# Now let's join it back together with a different character
word_list = ['Hello', 'World']
joined = '---'.join(word_list) # Puts '---' between each item in the list
print(joined)                  # Output: "Hello---World"
```

**4. Finding and Replacing**

```python
text = "Python is great, and Python is easy."

# .find() tells you the index where a substring starts (or -1 if not found)
position = text.find("great")
print(position)  # Output: 10

# .replace() returns a NEW string with replacements made
new_text = text.replace("Python", "Coding")
print(new_text)  # Output: "Coding is great, and Coding is easy."
```

**5. Checking the Content**
These methods are like asking the string a True/False question.

```python
print("abc".isalpha())   # Is this string all alphabetic characters? -> True
print("123".isdigit())   # Is this string all digits? -> True
print("User123".isalnum()) # Is this string all letters or numbers? -> True
```

#### Making Your Strings Dynamic with Formatting

Often, you'll want to insert variables into a string. The easiest and most modern way is with **f-strings**.

```python
# f-Strings (the best way, available in Python 3.6+)
# Just put an 'f' before the opening quote and variables in {curly_braces}.
name = "Bob"
age = 30
intro = f"Hi, my name is {name} and I am {age} years old."
print(intro) # Output: "Hi, my name is Bob and I am 30 years old."

# .format() method (You'll see this in older code)
intro_old = "{} is {} years old.".format(name, age)
```

#### Handling Special Characters

What if you need to put a quote inside a string that's already using those quotes? Use a backslash (`\`) to "escape" the character, telling Python to treat it as a literal character.

```python
# The \ tells Python that the " is part of the string, not the end of it.
# \n is a special escape character for a new line.
escaped = "She said, \"This is where the new line starts.\nSee?\""
print(escaped)
# Output:
# She said, "This is where the new line starts.
# See?"
```

By mastering these basic string operations, you're building a solid foundation for all the cool AI and data science projects to come. Next, you'll see how to use these skills to find even more complex patterns in text.
___

### Working with Text Files in Python

Alright, you know how to handle text *inside* your code. But in the real world, data doesn't just appear out of thin air—it lives in files. We're talking about simple `.txt` files (like a basic document from Notepad) or `.csv` files (think of a spreadsheet saved in a super simple format).

Python lets you open these files, read what's inside, make changes, and save your work. This is a crucial skill for almost any project, whether you're analyzing customer reviews from a `.csv` file or cleaning up a messy list of emails stored in a `.txt`.

**Why is this so important?** Because this is how you connect your code to real-world data. It’s how you build tools that can automate boring tasks or analyze huge datasets without crashing your computer. Let's break down how to do it, step by step.

**The Golden Rule: Always Use `with`**

Before we start, know this: the best way to open a file is with a `with` statement. Think of it as a safety net. It automatically closes the file for you when you're done, so you never have to worry about forgetting. Forgetting to close a file is like leaving a bunch of apps open on your phone—it hogs resources and can cause problems later.

#### **Step 1: Opening a File**

To work with a file, you use Python's built-in `open()` function. It needs two things:
1.  The file's name (e.g., `"myfile.txt"`).
2.  A "mode," which tells Python what you want to do.

The most common modes are:
*   `'r'` - **Read**: To look at what's inside the file. (This is the default if you forget to add a mode).
*   `'w'` - **Write**: To write new content to a file. **Warning:** This will erase everything that was already in the file!
*   `'a'` - **Append**: To add new content to the *end* of a file without deleting anything.

Here’s how you’d open a `.txt` file to read it:

```python
# This opens 'example.txt' in read mode.
# The file is automatically closed after the indented block.
with open('example.txt', 'r') as file:
    content = file.read()  # Reads the entire file into one big string
    print(content)
```

And here’s how you’d write to a file. If `newfile.txt` doesn't exist, Python will create it for you!

```python
with open('newfile.txt', 'w') as file:
    file.write("Hello, world!\n") # \n adds a new line
    file.write("This is the second line.")
```

Boom—you just created and wrote to a file with code!

#### **Step 2: Reading Files Without Crashing Your Computer**

Using `file.read()` is fine for small files. But what if your file is huge, like a 10 GB log file? Loading it all at once would eat up all your computer's memory and likely crash your program.

The smart way to handle big files is to read them **line by line**. Think of it like eating a giant pizza: you eat it slice by slice, not all at once.

This is the most common and memory-efficient way to read a file:

```python
# Let's clean up a big file by removing extra whitespace from each line
with open('bigfile.txt', 'r') as file:
    for line in file:  # This loop processes one line at a time
        clean_line = line.strip()  # .strip() removes leading/trailing spaces
        print(clean_line)
```
This method is a lifesaver. You can process files of any size because you only ever hold one line in memory at a time.

#### **Step 3: Reading and Writing `.csv` Files (The Spreadsheet Files)**

CSV (Comma-Separated Values) files are everywhere. They are just text files used to store table-like data, where commas separate the columns. For example: `Name,Age,City`.

Trying to split these lines yourself can get messy, especially if some of the data (like a movie title) contains a comma. Luckily, Python has a built-in `csv` module designed to handle this perfectly.

**Reading a `.csv` file:**

```python
import csv  # First, you need to import the csv toolkit

with open('data.csv', 'r') as file:
    reader = csv.reader(file)  # The reader turns each row into a list of strings
    
    # You can skip the header row if you have one
    header = next(reader)
    print(f"The columns are: {header}")

    # Now loop through the rest of the rows
    for row in reader:
        # Each 'row' is a list, like ['Alice', '25', 'New York']
        print(f"{row[0]} is {row[1]} years old and lives in {row[2]}.")
```

**Writing to a `.csv` file:**

```python
import csv

# Your data, organized as a list of lists
data_to_write = [
    ['Name', 'Major'], 
    ['Bob', 'Computer Science'], 
    ['Charlie', 'Biology']
]

# The newline='' is important! It prevents extra blank rows from being created.
with open('output.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(data_to_write)  # Writes all your data at once
```
This is perfect for when you need to analyze text from one column, clean it up, and save the results in a structured way.

#### **Tips for Success & Common Problems**

*   **"File Not Found" Error?** Make sure the file is in the same folder as your Python script, or provide the full path to it (e.g., `"C:/Users/YourName/Documents/file.txt"`).
*   **Don't Crash!** For any file that might be large, *always* read it line by line inside a loop.
*   **Practice Safely:** When you're learning, don't practice on important files! Create your own simple `test.txt` and `data.csv` files to experiment with.

File handling is the skill that turns your code from a simple script into a powerful tool. Now that you know how to connect your code to the real world, you can start building things that actually *do* something. 

---


### Working with PDF Files in Python

So, you've mastered plain `.txt` and `.csv` files. Now, let's talk about their fancy, complicated cousin: the **PDF**. PDFs are everywhere—think online textbooks, class syllabi, lab reports, and official forms.

But here's the catch: PDFs are designed to *look* good on any screen, not to be easily copied and pasted from. They can contain a mix of text, images, and complex layouts. Trying to grab text from them can feel like trying to unscramble a puzzle.

The good news? Python can help you unlock them! With the right library, you can turn a locked-down PDF into plain text that you can actually work with. This is a game-changer for projects like analyzing research papers, pulling data from reports, or even just grabbing the notes from a professor's slides.

#### Getting Your Toolkit: Installing `pypdf`

Python doesn't have a built-in tool for PDFs, so we need to install a free, open-source library. We'll use **`pypdf`** because it's powerful, beginner-friendly, and easy to set up.

First, you need to install it. Open your computer's terminal or command prompt and type this simple command:

```bash
pip install pypdf
```
**(Note: It's all lowercase! Just copy and paste that command and hit Enter.)**

Once it's installed, you're ready to start working with PDFs. Let's walk through the basics.

#### **Step 1: Opening and Exploring a PDF**

Before you can read the contents of a PDF, you need to open it and let Python take a look inside. We'll create a "reader" object, which is like giving Python a special magnifying glass to inspect the file.

Here’s how you can open a PDF and get some basic info about it, like the author or page count.

```python
# First, import the tool you need from the library
from pypdf import PdfReader

# Create a reader object by opening your PDF file in 'read binary' ('rb') mode
# Make sure 'example.pdf' is in the same folder as your script!
reader = PdfReader('example.pdf')

# You can get the number of pages
num_pages = len(reader.pages)
print(f"This PDF has {num_pages} pages.")

# You can also grab the "metadata" (info about the file)
info = reader.metadata
print(f"Author: {info.author}") # This will be None if no author is listed
```

This is great for quickly checking a file's properties without even opening it in a PDF viewer!

#### **Step 2: The Main Event: Extracting Text from a PDF**

This is the superpower you've been waiting for. You can loop through each page of a PDF and pull out all the text.

**A quick heads-up:** This works best on PDFs that were created from a text document (like saving a Word doc as a PDF). If the PDF is just a scan or a photo of a page, the computer sees a picture, not letters, and the extracted text might look like gibberish. We'll cover that in the "tips" section.

Here’s how to read the text from every page and combine it into a single string:

```python
from pypdf import PdfReader

reader = PdfReader('example.pdf')

full_text = "" # Create an empty string to hold all the text

# Loop through each page in the PDF
for page in reader.pages:
    # Extract the text from the current page and add it to our string
    full_text += page.extract_text() + "\n"

# Now you have one big string with all the text!
print(full_text)
```
Just like that, the PDF's content is now a normal Python string. You can use all the string methods you already know (`.lower()`, `.split()`, `.replace()`) to clean it up and analyze it.

If you only need the text from a specific page (like the first one), you can grab it directly:
`first_page_text = reader.pages[0].extract_text()` (Remember, Python starts counting from 0!)

#### **Step 3: Bonus Round - Merging PDFs**

Ever needed to combine several PDFs into one? Maybe for submitting a project or creating a study guide from different lecture notes? `pypdf` makes this incredibly simple.

```python
from pypdf import PdfMerger

# Create a merger object
merger = PdfMerger()

# Add the files you want to combine, one by one
merger.append('file1.pdf')
merger.append('file2.pdf')
# Add as many as you want!
# merger.append('file3.pdf')

# Write the final, merged PDF to a new file
merger.write('merged-document.pdf')

# Always close the merger to finalize the process
merger.close()
```
And voilà! You have a new file called `merged-document.pdf` with everything combined in order.

#### **Tips for Success & Common Hiccups**

*   **Gibberish Text?** If you extract text and get weird symbols, your PDF is likely an **image scan**. The computer can't "read" the letters. To solve this, you'd need a more advanced tool called OCR (Optical Character Recognition), but for now, just know that `pypdf` works best with text-based PDFs.
*   **"File Not Found" Error?** This is the most common error. Make sure your PDF file is saved in the *exact same folder* as your Python script.
*   **Practice Safely:** Don't experiment on your only copy of an important assignment! Download some free sample PDFs from the web to play with first.
*   **Connect it to What You Know:** Once you extract the text into a string, the real fun begins! Use the skills you've already learned: split the text into sentences, find keywords, or count word frequencies.

Processing PDFs feels like a secret power at first, but it’s a practical skill you’ll use again and again. Try it out with a syllabus or an article for one of your classes. 
___


### An Intro to Regular Expressions (Regex)

You know how to use `.find()` to search for a specific word. That's great, but what if your search is more complicated? What if you need to find *every email address* in a document, or *every phone number*, or *every hashtag* from a list of tweets?

That's where **Regular Expressions** (or **regex** for short) come in.

Think of regex as a "find and replace" on steroids. It's a special language for describing **patterns** in text. Instead of telling Python to "find the word 'cat'," you can tell it to "find me anything that looks like an email address." In text processing and NLP, this is a game-changing skill for cleaning data and extracting specific information.

Don't worry, it looks like a secret code at first, but it's more like learning a new mini-alphabet. Once you get the hang of it, you'll see patterns everywhere!

#### Getting Started with Python's `re` Module

Python has a built-in toolkit for regex called the `re` module, so there's nothing new to install. You just need to import it.

```python
import re # This brings in the regex tools
```
When we write a regex pattern, we'll almost always put an `r` in front of the string (like `r'your-pattern-here'`). This creates a "raw string," which helps Python avoid getting confused by special characters. Just trust us on this one—it's a best practice that will save you headaches later.

**The Regex Alphabet (The Basics)**

A regex pattern is a mix of normal characters (which match themselves) and special characters (which have superpowers).

*   `\d` - Matches any **d**igit (0-9).
*   `\w` - Matches any "**w**ord" character (letters, numbers, and the underscore).
*   `\s` - Matches any "**s**pace" character (spaces, tabs, newlines).
*   `.` - The "wildcard," matches any single character except a newline.
*   `+` - Matches "**one or more**" of the thing before it. (`\d+` means "one or more digits").
*   `*` - Matches "**zero or more**" of the thing before it.
*   `^` - Matches the **start** of the string.
*   `$` - Matches the **end** of the string.

Now, let's see how to use these in Python.

---

#### The 4 Most Important Regex Functions

The `re` module has a few key functions. Let's look at the four you'll use most often.

##### 1. `re.search()` - Is the pattern in here *anywhere*?

This function scans the entire string and stops as soon as it finds the *first* match.

```python
import re

text = "My favorite numbers are 19 and 42."
pattern = r'\d+'  # \d means digit, + means "one or more"

result = re.search(pattern, text)

if result:
    # .group() gives you the actual text that was found
    print(f"Found a match: {result.group()}")
else:
    print("No match found.")

# Output: Found a match: 19
```
Even though `42` is also a number, `re.search()` stops after finding the first match.

##### 2. `re.findall()` - Give me *all* of the matches!

This is one of the most useful functions. It finds every match in the string and returns them as a list.

```python
import re

text = "My contacts are alice@email.com and bob@work.net."
# A simple pattern for emails: word_chars + @ + word_chars + . + word_chars
pattern = r'\w+@\w+\.\w+'

emails = re.findall(pattern, text)
print(emails)

# Output: ['alice@email.com', 'bob@work.net']
```
Perfect for pulling out all the pieces of data that fit your pattern.

##### 3. `re.sub()` - Find and Replace

This function finds all matches and **sub**stitutes them with something else. It's incredibly powerful for cleaning or anonymizing data.

```python
import re

text = "Agent 47 will meet Agent 33 tomorrow."
pattern = r'\d+' # Find all numbers

# Replace every match with the string "[REDACTED]"
new_text = re.sub(pattern, "[REDACTED]", text)
print(new_text)

# Output: Agent [REDACTED] will meet Agent [REDACTED] tomorrow.
```

##### 4. `re.match()` - Does it match at the *very beginning*?

This one is more specific. It only checks for a match at the absolute start of the string. It's great for validating user input, like "Does this username start with a letter?"

```python
import re

text1 = "Username123"
text2 = "123Username"
pattern = r'\w+' # Does it start with a word character?

# Test on the first string
match1 = re.match(pattern, text1)
print(f"Checking '{text1}': {match1.group()}") # Output: Checking 'Username123': Username123

# Test on the second string
match2 = re.match(pattern, text2)
# This will cause an error if we don't check if a match was found first!
if match2:
    print(f"Checking '{text2}': {match2.group()}")
else:
    print(f"Checking '{text2}': No match at the beginning!")
# Output: Checking '123Username': 123Username
# Oh wait, let's make a better pattern for usernames starting with letters!
pattern = r'[a-zA-Z]\w*' # Starts with a letter, followed by zero or more word characters
match2 = re.match(pattern, text2)
if match2:
    print(f"Checking '{text2}' with new pattern: {match2.group()}")
else:
    print(f"Checking '{text2}' with new pattern: No match at the beginning!")
# Output: Checking '123Username' with new pattern: No match at the beginning!
```

---

#### Tips for Learning Regex

*   **Use an Online Tester!** The best way to learn is to play around. Websites like **regex101.com** are amazing. You can type in your pattern and your text, and it will highlight the matches and explain *why* it's matching in plain English.
*   **Start Simple.** Don't try to write a perfect, complex pattern on your first try. Build it up piece by piece.
*   **Know When to Use It.** If `text.replace()` or `text.find()` can do the job easily, just use them! Regex is for when the pattern is too complex for basic string methods.
*   **Combine Your Skills!** Remember how you extracted text from a PDF? Now you can use regex to search for patterns *inside* that extracted text.

Regex might feel like a puzzle at first, but it’s a skill that will make you a much more effective programmer. Give these examples a try—what's the first pattern you're going to hunt for?
