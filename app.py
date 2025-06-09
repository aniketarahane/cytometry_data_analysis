import streamlit as st
import pandas as pd
import sqlite3
from scipy.stats import mannwhitneyu
import plotly.express as px
import os

def create_schema(cursor):
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS projects (
        project_id INTEGER PRIMARY KEY,
        project_name TEXT UNIQUE NOT NULL
    );
    """)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS treatments (
        treatment_id INTEGER PRIMARY KEY,
        treatment_name TEXT UNIQUE NOT NULL
    );
    """)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS subjects (
        subject_id INTEGER PRIMARY KEY,
        subject_identifier TEXT UNIQUE NOT NULL,
        project_id INTEGER,
        condition TEXT,
        age INTEGER,
        sex TEXT,
        FOREIGN KEY (project_id) REFERENCES projects(project_id)
    );
    """)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS samples (
        sample_id INTEGER PRIMARY KEY,
        sample_identifier TEXT UNIQUE NOT NULL,
        subject_id INTEGER,
        treatment_id INTEGER,
        sample_type TEXT,
        response TEXT,
        time_from_treatment_start REAL,
        FOREIGN KEY (subject_id) REFERENCES subjects(subject_id),
        FOREIGN KEY (treatment_id) REFERENCES treatments(treatment_id)
    );
    """)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS cell_counts (
        count_id INTEGER PRIMARY KEY,
        sample_id INTEGER,
        b_cell INTEGER,
        cd8_t_cell INTEGER,
        cd4_t_cell INTEGER,
        nk_cell INTEGER,
        monocyte INTEGER,
        FOREIGN KEY (sample_id) REFERENCES samples(sample_id) ON DELETE CASCADE
    );
    """)

def populate_database(conn, df):
    cursor = conn.cursor()

    projects = df[['project']].drop_duplicates()
    for _, row in projects.iterrows():
        cursor.execute("INSERT OR IGNORE INTO projects (project_name) VALUES (?)", (row['project'],))

    treatments = df[['treatment']].drop_duplicates()
    for _, row in treatments.iterrows():
        cursor.execute("INSERT OR IGNORE INTO treatments (treatment_name) VALUES (?)", (row['treatment'],))

    subjects = df[['subject', 'project', 'condition', 'age', 'sex']].drop_duplicates('subject')
    for _, row in subjects.iterrows():
        cursor.execute("SELECT project_id FROM projects WHERE project_name = ?", (row['project'],))
        project_id = cursor.fetchone()[0]
        cursor.execute("""
            INSERT OR IGNORE INTO subjects (subject_identifier, project_id, condition, age, sex)
            VALUES (?, ?, ?, ?, ?)
        """, (row['subject'], project_id, row['condition'], row['age'], row['sex']))

    for _, row in df.iterrows():
        cursor.execute("SELECT subject_id FROM subjects WHERE subject_identifier = ?", (row['subject'],))
        subject_id = cursor.fetchone()[0]
        cursor.execute("SELECT treatment_id FROM treatments WHERE treatment_name = ?", (row['treatment'],))
        treatment_id = cursor.fetchone()[0]

        cursor.execute("""
            INSERT OR IGNORE INTO samples (sample_identifier, subject_id, treatment_id, sample_type, response, time_from_treatment_start)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (row['sample'], subject_id, treatment_id, row['sample_type'], row['response'], row['time_from_treatment_start']))

        cursor.execute("SELECT sample_id FROM samples WHERE sample_identifier = ?", (row['sample'],))
        sample_id = cursor.fetchone()[0]

        cursor.execute("""
            INSERT OR IGNORE INTO cell_counts (sample_id, b_cell, cd8_t_cell, cd4_t_cell, nk_cell, monocyte)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (sample_id, row['b_cell'], row['cd8_t_cell'], row['cd4_t_cell'], row['nk_cell'], row['monocyte']))

    conn.commit()

def init_db(db_file='cell_data.db', csv_file='cell-count.csv'):
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='samples'")
    table_exists = cursor.fetchone()

    if not table_exists:
        try:
            df = pd.read_csv(csv_file)
            create_schema(cursor)
            populate_database(conn, df)
        except FileNotFoundError:
            st.error(f"Error: '{csv_file}' not found. Please make sure you have uploaded it.")
            return None
        except Exception as e:
            st.error(f"An error occurred during initial database setup: {e}")
            conn.close()
            if os.path.exists(db_file):
                os.remove(db_file)
            return None
    return conn

def add_sample(conn, data):
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT OR IGNORE INTO projects (project_name) VALUES (?)", (data['project'],))
        cursor.execute("SELECT project_id FROM projects WHERE project_name = ?", (data['project'],))
        project_id = cursor.fetchone()[0]

        cursor.execute("INSERT OR IGNORE INTO treatments (treatment_name) VALUES (?)", (data['treatment'],))
        cursor.execute("SELECT treatment_id FROM treatments WHERE treatment_name = ?", (data['treatment'],))
        treatment_id = cursor.fetchone()[0]

        cursor.execute("INSERT OR IGNORE INTO subjects (subject_identifier, project_id, condition, age, sex) VALUES (?, ?, ?, ?, ?)",
                       (data['subject'], project_id, data['condition'], data['age'], data['sex']))
        cursor.execute("SELECT subject_id FROM subjects WHERE subject_identifier = ?", (data['subject'],))
        subject_id = cursor.fetchone()[0]

        cursor.execute("INSERT INTO samples (sample_identifier, subject_id, treatment_id, sample_type, response, time_from_treatment_start) VALUES (?, ?, ?, ?, ?, ?)",
                       (data['sample'], subject_id, treatment_id, data['sample_type'], data['response'], data['time_from_treatment_start']))
        sample_id = cursor.lastrowid

        cursor.execute("INSERT INTO cell_counts (sample_id, b_cell, cd8_t_cell, cd4_t_cell, nk_cell, monocyte) VALUES (?, ?, ?, ?, ?, ?)",
                       (sample_id, data['b_cell'], data['cd8_t_cell'], data['cd4_t_cell'], data['nk_cell'], data['monocyte']))

        conn.commit()
        return True
    except sqlite3.IntegrityError:
        st.error(f"Error: Sample identifier '{data['sample']}' already exists. Please use a unique identifier.")
        return False
    except Exception as e:
        st.error(f"An error occurred while adding the sample: {e}")
        return False

def get_full_data(conn):
    query = """
    SELECT
        p.project_name as project,
        su.subject_identifier as subject,
        su.condition,
        su.age,
        su.sex,
        t.treatment_name as treatment,
        s.response,
        s.sample_identifier as sample,
        s.sample_type,
        s.time_from_treatment_start,
        cc.b_cell,
        cc.cd8_t_cell,
        cc.cd4_t_cell,
        cc.nk_cell,
        cc.monocyte
    FROM samples s
    JOIN subjects su ON s.subject_id = su.subject_id
    JOIN projects p ON su.project_id = p.project_id
    JOIN treatments t ON s.treatment_id = t.treatment_id
    JOIN cell_counts cc ON s.sample_id = cc.sample_id
    """
    return pd.read_sql(query, conn)

def main():
    st.set_page_config(layout="wide")
    st.title("Cell Population Analysis Dashboard")

    if 'message' in st.session_state:
        st.success(st.session_state.pop('message'))

    conn = init_db()
    if conn is None:
        st.stop()

    st.sidebar.header("Data Management")
    with st.sidebar.expander("Add or Remove Sample", expanded=False):
        st.subheader("Add a New Sample")
        with st.form("add_sample_form", clear_on_submit=True):
            project = st.text_input("Project ID")
            subject = st.text_input("Subject ID")
            condition = st.selectbox("Condition", ["melanoma", "healthy", "other"])
            age = st.number_input("Age", min_value=0, max_value=120, step=1)
            sex = st.selectbox("Sex", ["F", "M"])
            treatment = st.text_input("Treatment")
            response = st.selectbox("Response", ["y", "n", "NaN"], format_func=lambda x: "Not Applicable" if x == "NaN" else x)
            sample = st.text_input("Sample ID (must be unique)")
            sample_type = st.selectbox("Sample Type", ["PBMC", "Tissue"])
            time_from_treatment_start = st.number_input("Time From Treatment Start (days)", value=0.0)
            b_cell = st.number_input("B Cell Count", min_value=0, step=100)
            cd8_t_cell = st.number_input("CD8+ T-Cell Count", min_value=0, step=100)
            cd4_t_cell = st.number_input("CD4+ T-Cell Count", min_value=0, step=100)
            nk_cell = st.number_input("NK Cell Count", min_value=0, step=100)
            monocyte = st.number_input("Monocyte Count", min_value=0, step=100)

            submitted = st.form_submit_button("Add Sample to Database")
            if submitted:
                sample_data = {
                    "project": project, "subject": subject, "condition": condition, "age": age, "sex": sex,
                    "treatment": treatment, "response": None if response == 'NaN' else response, "sample": sample, "sample_type": sample_type,
                    "time_from_treatment_start": time_from_treatment_start, "b_cell": b_cell,
                    "cd8_t_cell": cd8_t_cell, "cd4_t_cell": cd4_t_cell, "nk_cell": nk_cell, "monocyte": monocyte
                }
                if add_sample(conn, sample_data):
                    st.session_state.message = f"Sample '{sample}' added successfully!"
                    st.rerun()

        st.subheader("Remove a Sample")
        all_samples_df = pd.read_sql("SELECT sample_identifier FROM samples", conn)
        if not all_samples_df.empty:
            sample_to_remove = st.selectbox("Select sample to remove", all_samples_df['sample_identifier'].unique(), key="remove_selectbox")
            if st.button("Remove Selected Sample"):
                cursor = conn.cursor()
                cursor.execute("DELETE FROM samples WHERE sample_identifier = ?", (sample_to_remove,))
                conn.commit()
                st.session_state.message = f"Sample '{sample_to_remove}' has been removed."
                st.rerun()

    db_df = get_full_data(conn)

    col1, col2 = st.columns(2)

    with col1:
        st.header("Sample Summary")
        available_samples = db_df['sample'].unique()
        if len(available_samples) > 0:
            sample_to_analyze = st.selectbox("Select a sample:", available_samples)

            if sample_to_analyze:
                sample_df_check = db_df[db_df['sample'] == sample_to_analyze]
                if not sample_df_check.empty:
                    sample_data = sample_df_check.iloc[0]
                    cell_populations = ['b_cell', 'cd8_t_cell', 'cd4_t_cell', 'nk_cell', 'monocyte']
                    total_cells = sample_data[cell_populations].sum()
                    summary_data = [{"Cell Population": pop.replace('_', ' ').title(), "Count": f"{int(sample_data[pop]):,}", "Relative Frequency": f"{(sample_data[pop] / total_cells) * 100:.2f}%"} for pop in cell_populations]
                    st.subheader(f"Summary Table for Sample {sample_to_analyze}")
                    st.table(pd.DataFrame(summary_data))
        else:
            st.warning("No samples available for analysis. Please add a sample.")

        st.header("Melanoma PBMC Baseline Samples with Treatment 1 Summary")
        baseline_df = db_df[(db_df['condition'] == 'melanoma') & (db_df['sample_type'] == 'PBMC') & (db_df['treatment'] == 'tr1') & (db_df['time_from_treatment_start'] == 0)]
        if not baseline_df.empty:
            st.markdown(f"""
            - **Total Baseline Samples:** {len(baseline_df)}
            - **Samples by Project:** {', '.join([f'{k}: {v}' for k, v in baseline_df['project'].value_counts().to_dict().items()])}
            - **Responders (y) vs. Non-responders (n):** {baseline_df['response'].value_counts().get('y', 0)} vs. {baseline_df['response'].value_counts().get('n', 0)}
            - **Female vs. Male Subjects:** {baseline_df['sex'].value_counts().get('F', 0)} vs. {baseline_df['sex'].value_counts().get('M', 0)}
            """)
        else:
            st.warning("No baseline Melanoma PBMC samples with TR1 treatment found.")

    with col2:
        st.header("Melanoma Patient Treatment 1 Responders vs Non-Responders")
        melanoma_tr1_pbmc = db_df[(db_df['condition'] == 'melanoma') & (db_df['treatment'] == 'tr1') & (db_df['sample_type'] == 'PBMC') & (db_df['response'].isin(['y', 'n']))].copy()
        if not melanoma_tr1_pbmc.empty:
            cell_populations = ['b_cell', 'cd8_t_cell', 'cd4_t_cell', 'nk_cell', 'monocyte']
            melanoma_tr1_pbmc['total_cells'] = melanoma_tr1_pbmc[cell_populations].sum(axis=1)
            for pop in cell_populations:
                melanoma_tr1_pbmc[f'{pop}_rel'] = (melanoma_tr1_pbmc[pop] / melanoma_tr1_pbmc['total_cells']) * 100

            st.subheader("Average Cell Population Frequencies by Response")
            comparison_table = melanoma_tr1_pbmc.groupby('response')[[f'{pop}_rel' for pop in cell_populations]].mean().T
            comparison_table.columns = ['Non-responders (n)', 'Responders (y)']
            comparison_table.index = [name.replace('_rel', '').replace('_', ' ').title() for name in comparison_table.index]
            st.table(comparison_table.style.format("{:.2f}%"))

            st.subheader("Distributions of Cell Population Frequencies")
            melted_df = melanoma_tr1_pbmc.melt(id_vars=['response'], value_vars=[f'{pop}_rel' for pop in cell_populations], var_name='Cell Population', value_name='Relative Frequency (%)')
            melted_df['Cell Population'] = melted_df['Cell Population'].str.replace('_rel', '').str.replace('_', ' ').str.title()
            fig = px.box(melted_df, x='Cell Population', y='Relative Frequency (%)', color='response', title="Responders vs. Non-responders", labels={"response": "Response"}, color_discrete_map={'y': 'royalblue', 'n': 'firebrick'})
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Statistical Significance (Mann-Whitney U Test)")
            responders = melanoma_tr1_pbmc[melanoma_tr1_pbmc['response'] == 'y']
            non_responders = melanoma_tr1_pbmc[melanoma_tr1_pbmc['response'] == 'n']
            stats_results = []
            for pop in cell_populations:
                if not responders[f'{pop}_rel'].empty and not non_responders[f'{pop}_rel'].empty:
                    stat, p_value = mannwhitneyu(responders[f'{pop}_rel'].dropna(), non_responders[f'{pop}_rel'].dropna(), alternative='two-sided')
                    stats_results.append({'Cell Population': pop.replace('_', ' ').title(), 'p-value': p_value, 'Significant (p < 0.05)': 'Yes' if p_value < 0.05 else 'No'})
            if stats_results:
                st.table(pd.DataFrame(stats_results).style.format({'p-value': '{:.4f}'}))
            else:
                st.info("Not enough data to perform statistical tests.")
        else:
            st.warning("No data available to compare TR1 responders and non-responders.")

    conn.close()

if __name__ == "__main__":
    main()
