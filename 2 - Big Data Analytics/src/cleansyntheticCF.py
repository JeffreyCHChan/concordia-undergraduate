def clean_synthetic_cf():
    courseList = []
    with open("content_based_student_courses", 'r') as file:
        for line in file:
            line = line.split("::")
            courseName = line[2]
            courseList.append(courseName)
    with open("content_based_student_courses_extracted", 'w') as f:
        f.write('\n'.join(courseList))

clean_synthetic_cf()

