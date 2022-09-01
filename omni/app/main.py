# import requirements needed
from flask import Flask, render_template, request, render_template_string
from utils import get_base_url
import pandas as pd
import plotly.express as px
import plotly.io as pi
import os
import plotly.graph_objects as go

import numpy as np
import matplotlib.pyplot as plt
import statistics

# importing datasets into variables
import pandas as pd
import plotly.express as px

# :)


women_LF = pd.read_csv("./Women in Labor Force.csv")

women_LF = women_LF.drop(["Indicator Name", "Indicator Code",'1960', '1961', '1962', '1963', '1964', '1965', '1966',
       '1967', '1968', '1969', '1970', '1971', '1972', '1973', '1974', '1975',
       '1976', '1977', '1978', '1979', '1980', '1981', '1982', '1983', '1984',
       '1985', '1986', '1987', '1988', '1989', 'Unnamed: 66'], axis = 1)

africa = pd.read_csv("./Data -- World Continents /Africa population.csv")
europe = pd.read_csv("./Data -- World Continents /europe population.csv")
nAmerica = pd.read_csv("./Data -- World Continents /north america population.csv")
sAmerica = pd.read_csv("./Data -- World Continents /south america population.csv")
asia = pd.read_csv("./Data -- World Continents /Asia population.csv")
australia = pd.read_csv("./Data -- World Continents /Australia population.csv")

# Converting obj/string into float 
def toInt(dataframe, column):
    dataframe[column] = dataframe.apply(lambda row : float(row.astype(str)[column].replace(",", "")), axis=1)

toInt(africa, "World Population")
toInt(asia, "World Population")
toInt(australia, "World Population")
toInt(nAmerica, "World Population")
toInt(sAmerica, "World Population")
toInt(europe, "World Population")

# Apending landsize into dataset (land size of each continent as of today (km^2))
africa["Continent"] = "Africa"
africa["Land Size"] = 30065000
europe["Continent"] = "Europe"
europe["Land Size"] = 9938000
asia["Continent"] = "Asia"
asia["Land Size"] = 44579000
australia["Continent"] = "Australia"
australia["Land Size"] = 7687000
nAmerica["Continent"] = "North America"
nAmerica["Land Size"] = 24256000
sAmerica["Continent"] = "South America"
sAmerica["Land Size"] = 17819000

#combining datasets into one dataset (populations)
new_df = africa.append(europe)
new_df2 = new_df.append(asia)
new_df3 = new_df2.append(australia)
new_df4 = new_df3.append(nAmerica)
populations = new_df4.append(sAmerica)

populations = populations.drop(["Africa's Rank in World Population", "Africa's Share of World Population", "Asia's Rank in World Population", "Asia's Share of World Population", "Australia's Rank in World Population", "Australia's Share of World Population", "South America's Rank in World Population", "South America's Share of World Population", "North America's Rank in World Population", "North America's Share of World Population", "Europe's Rank in World Population", "Europe's Share of World Population"], axis = 1)


dataset = pd.read_excel("./World Population.xlsx")
bangladesh = pd.read_excel("./World Population.xlsx", sheet_name="Bangladesh")
brazil = pd.read_excel("./World Population.xlsx", sheet_name="Brazil")
china = pd.read_excel("./World Population.xlsx", sheet_name="China")
dr_congo = pd.read_excel("./World Population.xlsx", sheet_name="DR Congo")
egypt = pd.read_excel("./World Population.xlsx", sheet_name="Egypt")
ethiopia = pd.read_excel("./World Population.xlsx", sheet_name="Ethiopia")
france = pd.read_excel("./World Population.xlsx", sheet_name="France")
germany = pd.read_excel("./World Population.xlsx", sheet_name="Germany")
india = pd.read_excel("./World Population.xlsx", sheet_name="India")
indonesia = pd.read_excel("./World Population.xlsx", sheet_name="Indonasia")
iran = pd.read_excel("./World Population.xlsx", sheet_name="Iran")
italy = pd.read_excel("./World Population.xlsx", sheet_name="Italy")
japan = pd.read_excel("./World Population.xlsx", sheet_name="Japan")
mexico = pd.read_excel("./World Population.xlsx", sheet_name="Mexico")
nigeria = pd.read_excel("./World Population.xlsx", sheet_name="Nigeria")
pakistan = pd.read_excel("./World Population.xlsx", sheet_name="Pakistan")
philippines = pd.read_excel("./World Population.xlsx", sheet_name="Philippines")
russia = pd.read_excel("./World Population.xlsx", sheet_name="Russia")
south_africa = pd.read_excel("./World Population.xlsx", sheet_name="South Africa")
tanzania = pd.read_excel("./World Population.xlsx", sheet_name="Tanzania")
thailand = pd.read_excel("./World Population.xlsx", sheet_name="Thailand")
turkey = pd.read_excel("./World Population.xlsx", sheet_name="Turkey")
usa = pd.read_excel("./World Population.xlsx", sheet_name = "USA")
uk = pd.read_excel("./World Population.xlsx", sheet_name = "United Kingdom")
vietnam = pd.read_excel("./World Population.xlsx", sheet_name="Vietnam")


def render_visual2(year):
    print(params)
    

# setup the webserver
# port may need to be changed if there are multiple flask servers running on same server
port = 12222
base_url = get_base_url(port)


# if the base url is not empty, then the server is running in development, and we need to specify the static folder so that the static files are served
if base_url == '/':
    app = Flask(__name__)
else:
    app = Flask(__name__, static_url_path=base_url + 'static')


# set up the routes and logic for the webserver
@app.route(f'{base_url}')
def home():
    return render_template('index.html')


# define additional routes here
# for example:
# @app.route(f'{base_url}/team_members')
# def team_members():
#     return render_template('team_members.html') # would need to actually make this page


@app.route(f'{base_url}/visuals')
def visuals():
    return render_template('visuals.html')


@app.route(f'{base_url}/blog-post')
def blogpost():
    return render_template('blog-post.html')


@app.route(f'{base_url}/about')
def about():
    return render_template('about.html')


@app.route(f'{base_url}/machine-learning')
def machinelearning():
    return render_template('machine-learning.html')


@app.route(f'{base_url}/faq')
def faq():
    return render_template('faq.html')


@app.route(f'{base_url}/visuals', methods=['POST'])
@app.route(f'{base_url}/machine-learning', methods=['POST'])
def my_form_post1():
    print(request.url)
    if "visuals" in request.url:
        form = request.form
        if 'col1' in form.keys():
            dependent = str(form['col1'])
            relation = str(form['col2'])
            print(populations.columns)
            print(relation)
            print(dependent)
            
            factors = ["Population", "Median Age", "Fertility Rate", "Density (P/Km)", "Urban Population"]
            index = 0
            while (dependent == relation):
                dependent = factors[index]
                index = index + 1
            fig3 = px.scatter(populations[["Year", dependent, "Continent", relation]].dropna(), x="Year", y=dependent, title="The " + dependent + " Amongst the Continents Over Time in Relation to " + relation, color="Continent", opacity=0.8 , size=relation)
            try:
                os.remove("./templates/fig3.html")
            except:
                print('file doesnt exist, vis1')
                
            fig3.write_html("./templates/fig3.html")
            
            
            print('first form submit')
            return render_template('visuals.html')
        
        
        elif 'year1' in form.keys():
            year = form['year1']
            
            choro = go.Figure(data=go.Choropleth(
            locations = women_LF['Country Code'],
            z = women_LF[year],
            text = women_LF['Country Name'],
            colorscale = 'sunset',
            autocolorscale=False,
            reversescale=False,
            marker_line_color='darkgray',
            marker_line_width=0.5,
            colorbar_tickprefix = '%',
            colorbar_title = 'Female Labor Force Participation Rate',
            ))

            choro.update_layout(
            title_text='Female Labor Force Participation Rate',
            geo=dict(
                showframe=False,
                showcoastlines=False,
                projection_type='equirectangular'
                ),
            #annotations = [dict(
                #x=0.55,
                #y=0.1,
                #xref='paper',
                #yref='paper',
                #text='Source: <a href="https://www.cia.gov/library/publications/the-world-factbook/fields/2195.html">\
                   # CIA World Factbook</a>',
                #showarrow = False
           # )]
            )
            
            try:
                os.remove("./templates/choro.html")
            except:
                print('file doesnt exist, vis2')
                
            choro.write_html("./templates/choro.html")
            
            return render_template('visuals.html')
        
        elif 'country1' in form.keys():
            country = form['country1']
            factor1 = form['col2']
            factor2 = form['col3']
            
            print(country, factor1, factor2)
            country_dict= {"bangladesh" : bangladesh, "brazil": brazil, "china": china, "dr_congo" : dr_congo, "egypt" : egypt, "ethiopia" : ethiopia, "france" : france, "germany" : germany, "india": india, "indonesia" : indonesia, "iran": iran, "italy" : italy, "japan" : japan, "mexico" : mexico, "nigeria" : nigeria, "pakistan" : pakistan, "philippines" : philippines, "russia" : russia, "south_africa" : south_africa, "tanzania": tanzania, "thailand": thailand, "turkey" : turkey, "usa" : usa, "uk" : uk, "vietnam" : vietnam}

            #toInt(country_dict[country], col1)
            #toInt(country_dict[country], col2)
            country_dict[country][factor1].astype(float).dropna()
            country_dict[country][factor2].astype(float).dropna()
            print(country_dict[country].columns)
            fig1 = px.scatter_3d(country_dict[country], x= "Year", y= factor1, z=factor2,
                          color = factor1, size = factor2, title = country)
                        
            try:
                os.remove("./templates/fig1.html")
            except:
                print('file doesnt exist, vis2')
                
            fig1.write_html("./templates/fig1.html")
            
            print('third submit')
        

        return render_template('visuals.html')
    
    elif "machine-learning" in request.url:
        print('haha')
        form = request.form
        if 'country2' in form.keys():
            country = form['country2']
            xaxis = form['col4']
            yaxis = form['col5']

            df = pd.read_excel("./World Population.xlsx", sheet_name=country)

            X = df[xaxis].to_numpy()
            y = df[yaxis].to_numpy()

            length = len(usa.index)

            idx = np.arange(length)
            np.random.shuffle(idx)

            split_threshold = int(length * 0.8)

            train_idx = idx[:split_threshold]
            test_idx = idx[split_threshold:]

            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]
            X_train = X_train.reshape(-1, 1)
            y_train = y_train.reshape(-1, 1)
            X_test = X_test.reshape(-1, 1)

            X = X.reshape(-1, 1)
            y = y.reshape(-1, 1)

            kernel = gp.kernels.ConstantKernel(1.0, (1e-1, 1e3)) * gp.kernels.RBF(10.0, (1e-3, 1e3))

            model = gp.GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=0.1, normalize_y=True)

            model.fit(X_train, y_train)
            params = model.kernel_.get_params()

            y_pred, std = model.predict(X_test, return_std=True)

            mean_prediction, std_prediction = model.predict(X, return_std=True)

            plt.plot(X, y, label=r"$f(x)$", linestyle="dotted")
            plt.scatter(X_train, y_train, label="Observations")
            plt.plot(X, mean_prediction, label="Mean prediction")
            plt.fill_between(
                X.ravel(),
                mean_prediction.T[0] - 1.96 * std_prediction,
                mean_prediction.T[0] + 1.96 * std_prediction,
                alpha=0.5,
                label=r"95% confidence interval",
            )
            plt.legend()
            plt.xlabel(xaxis)
            plt.ylabel(yaxis)
            _ = plt.title("Gaussian process regression on population dataset")
            
        return render_template("machine-learning.html", result=out)


@app.route(f'{base_url}/vis2')
def vis2():
    return render_template('choro.html')


@app.route(f'{base_url}/vis1')
def vis1():
    return render_template('fig57.html')

@app.route(f'{base_url}/render_fig3')
def vis3():
    return render_template('fig1.html')

@app.route(f'{base_url}/render_fig2')
def render_fig2():
    return render_template('t1_user_fig2.html')
        
    

@app.route(f'{base_url}/render_result')
def render_result():
    i = 2
    return render_template_string(f"<html><body><p>2</p></body></html>")
    
if __name__ == '__main__':
    # IMPORTANT: change url to the site where you are editing this file.
    website_url = 'cocalc16.ai-camp.dev'

    print(f'Try to open\n\n    https://{website_url}' + base_url + '\n\n')
    app.run(host='0.0.0.0', port=port, debug=True)