# **Cell Population Analysis Dashboard**

## **Code Overview**

* The data handling functions such as `create_schema`, `populate_database`, `add_sample`, and so on are mainly just to set up the database according to the schema that I thought was best.  
* I decided to use a Streamlit app to create a dashboard because it’s simple, quick, and powerful.

## **Schema and Scaling**

I designed the database with five tables(normalization) to make sure there isn’t repeating information, so as we add more and more entries, it does so in a way that minimizes redundancy.

#### **Schema Outline:**

* **`projects`**: Stores a unique list of all project names; this also allows to add more information about projects later  
* **`subjects`**: Stores subject details which links to a project.  
* **`treatments`**: Stores a unique list of all treatment names; this also allows to add more information about treatments later  
* **`samples`**: Stores sample details which links to a subject and a treatment.  
* **`cell_counts`**: Stores the actual cell measurements, linking to a sample.

## **Requirements**

Python 3.7+ and the following libraries:

* `streamlit`  
* `pandas`  
* `scipy`  
* `plotly`

## **Setup Instructions**

### **1\. Create a Project Directory**

mkdir cell\_analysis\_dashboard  
cd cell\_analysis\_dashboard

### **2\. Add Project Files**

Place the following two files inside the `cell_analysis_dashboard` folder:

1. `app.py`  
2. `cell-count.csv`

### **3\. Set Up a Virtual Environment**

\# For Mac/Linux  
python3 \-m venv venv  
source venv/bin/activate

\# For Windows  
python \-m venv venv  
.\\venv\\Scripts\\activate

### **4\. Install Dependencies**

pip install streamlit pandas scipy plotly

## **Running the Dashboard**

Once the setup is complete, run the following command in your terminal:

streamlit run app.py

