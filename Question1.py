def count_vowels_consonants(text):
    vowels = "aeiouAEIOU"
    v = 0
    c = 0

    for ch in text:
        if ch.isalpha():
            if ch in vowels:
                v += 1
            else:
                c += 1

    return v, c


# Main program
string_input = "ComputerScience"
vowels, consonants = count_vowels_consonants(string_input)
print("Vowels:", vowels)
print("Consonants:", consonants)
