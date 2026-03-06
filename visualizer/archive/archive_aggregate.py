    # ----------------------------------------
    # ARCHIVE: AGGREGATOR PAGE FUNCTIONS
    # ----------------------------------------
    
            self.app.post("/aggregate-plot")(self.get_aggregate_plot)
        self.app.post("/prices")(self.receive_prices)


    async def get_aggregate_data(self, request: DataRequest):
        try:
            error = self.check_request(request, aggregate=True)
            if error:
                print(error)
                return error
            
            self.data[request] = {}
            self.timestamp_min_max[request] = {}
            async with self.AsyncSessionLocal() as session:
                import time
                query_start = time.time()
                print("Querying journaldb...")
                stmt = select(MessageSql).filter(
                    MessageSql.from_alias == f"hw1.isone.me.versant.keene.{request.house_alias}.scada",
                    or_(
                        MessageSql.message_type_name == "batched.readings",
                        MessageSql.message_type_name == "report",
                        MessageSql.message_type_name == "snapshot.spaceheat",
                    ),
                    MessageSql.message_persisted_ms >= request.start_ms,
                    MessageSql.message_persisted_ms <= request.end_ms + 10*60*1000,
                ).order_by(asc(MessageSql.message_persisted_ms))

                result = await session.execute(stmt)
                all_raw_messages: List[MessageSql] = result.scalars().all()
                print(f"Time to fetch data: {round(time.time()-query_start,1)}s")

            if not all_raw_messages:
                warning_message = f"No data found for the aggregation in the selected timeframe."
                return {"success": False, "message": warning_message, "reload": False}
            
            for house_alias in set([message.from_alias for message in all_raw_messages]):
                if 'maple' in house_alias:
                    print("Skipped maple")
                    continue
                self.data[request][house_alias] = {}

                # Process reports
                reports: List[MessageSql] = sorted([
                    x for x in all_raw_messages 
                    if x.message_type_name in ['report', 'batched.readings']
                    and x.from_alias == house_alias
                    ], key = lambda x: x.message_persisted_ms
                    )
                self.data[request][house_alias] = {}
                for message in reports:
                    for channel in message.payload['ChannelReadingList']:
                        if message.message_type_name == 'report':
                            channel_name = channel['ChannelName']
                        elif message.message_type_name == 'batched.readings':
                            for dc in message.payload['DataChannelList']:
                                if dc['Id'] == channel['ChannelId']:
                                    channel_name = dc['Name']
                        if not channel['ValueList'] or not channel['ScadaReadTimeUnixMsList']:
                            continue
                        if len(channel['ValueList'])!=len(channel['ScadaReadTimeUnixMsList']):
                            continue
                        if ((channel_name not in ['hp-idu-pwr', 'hp-odu-pwr'] and 'depth' not in channel_name) 
                            or 'micro' in channel_name):
                            continue
                        if channel_name not in self.data[request][house_alias]:
                            self.data[request][house_alias][channel_name] = {'values': [], 'times': []}
                        self.data[request][house_alias][channel_name]['values'].extend(channel['ValueList'])
                        self.data[request][house_alias][channel_name]['times'].extend(channel['ScadaReadTimeUnixMsList'])
                if not self.data[request][house_alias]:
                    print(f"No data found for {house_alias}")
                    continue

                # Process snapshots
                max_timestamp = max(max(self.data[request][house_alias][channel_name]['times']) for channel_name in self.data[request][house_alias])
                snapshots = sorted(
                        [x for x in all_raw_messages if x.message_type_name=='snapshot.spaceheat'
                        and x.message_persisted_ms >= max_timestamp], 
                        key = lambda x: x.message_persisted_ms
                        )
                for snapshot in snapshots:
                    for snap in snapshot.payload['LatestReadingList']:
                        if snap['ChannelName'] in self.data[request][house_alias]:
                            self.data[request][house_alias][snap['ChannelName']]['times'].append(snap['ScadaReadTimeUnixMs'])
                            self.data[request][house_alias][snap['ChannelName']]['values'].append(snap['Value'])

                # Get minimum and maximum timestamp for plots
                max_timestamp = max(max(self.data[request][house_alias][x]['times']) for x in self.data[request][house_alias])
                min_timestamp = min(min(self.data[request][house_alias][x]['times']) for x in self.data[request][house_alias])
                min_timestamp += -(max_timestamp-min_timestamp)*0.05
                max_timestamp += (max_timestamp-min_timestamp)*0.05
                self.timestamp_min_max[request][house_alias] = {
                    'min_timestamp': self.to_datetime(min_timestamp),
                    'max_timestamp': self.to_datetime(max_timestamp+5*60*60*1000)
                }

                # Sort values according to time and convert to datetime
                for channel_name in self.data[request][house_alias].keys():
                    sorted_times_values = sorted(zip(self.data[request][house_alias][channel_name]['times'], self.data[request][house_alias][channel_name]['values']))
                    sorted_times, sorted_values = zip(*sorted_times_values)
                    self.data[request][house_alias][channel_name]['values'] = list(sorted_values)
                    self.data[request][house_alias][channel_name]['times'] = pd.to_datetime(list(sorted_times), unit='ms', utc=True)
                    self.data[request][house_alias][channel_name]['times'] = self.data[request][house_alias][channel_name]['times'].tz_convert(self.timezone_str)
                    self.data[request][house_alias][channel_name]['times'] = [x.replace(tzinfo=None) for x in self.data[request][house_alias][channel_name]['times']]        
                
            # Re-sample to equal timesteps
            print("Re-sampling...")
            start_ms = request.start_ms
            end_ms = request.end_ms + (10*60*1000 if query_start-request.end_ms/1000>10*60 else 0)
            timestep_s = 30
            num_points = int((end_ms - start_ms) / (timestep_s * 1000) + 1)
            sampling_times = np.linspace(start_ms, end_ms, num_points)
            sampling_times = pd.to_datetime(sampling_times, unit='ms', utc=True)
            sampling_times = [x.tz_convert(self.timezone_str).replace(tzinfo=None) for x in sampling_times]

            agg_data = {}
            for house_alias in self.data[request]:
                agg_data[house_alias] = {'timestamps': sampling_times}
                for channel in self.data[request][house_alias]:
                    sampled = await asyncio.to_thread(
                        pd.merge_asof, 
                        pd.DataFrame({'times': sampling_times}),
                        pd.DataFrame(self.data[request][house_alias][channel]),
                        on='times', 
                        direction='backward'
                        )
                    sampled['values'] = sampled['values'].bfill()
                    agg_data[house_alias][channel] = list(sampled['values'])

                # Compute average temperature and energy
                temperature_channels = [value for key, value in agg_data[house_alias].items() if 'depth' in key]
                num_lists = len(temperature_channels)
                num_elements = len(temperature_channels[0])
                sums = [0] * num_elements
                for channel in temperature_channels:
                    for i in range(num_elements):
                        sums[i] += channel[i]
                averaged_temperature = [sum_value / num_lists for sum_value in sums]
                m_total_kg = 120*4*3.785
                agg_data[house_alias]['energy'] = [m_total_kg*4.187/3600*(avg_temp/1000-30) for avg_temp in averaged_temperature]
                agg_data[house_alias] = {k: v for k, v in agg_data[house_alias].items() if 'depth' not in k}
            
            energy_list, hp_list = [], []
            for i in range(len(agg_data[house_alias]['energy'])):
                energy_list.append(sum([agg_data[ha]['energy'][i] for ha in agg_data]))
                hp_list.append(sum([(agg_data[ha]['hp-idu-pwr'][i]+agg_data[ha]['hp-odu-pwr'][i])/1000 for ha in agg_data]))
            # Remove the last minutes of the energy plot to avoid wierd behaviour
            energy_list = [
                energy if t<datetime.fromtimestamp(end_ms/1000-10*60,pytz.timezone(self.timezone_str)).replace(tzinfo=None) else np.nan
                for t, energy in zip(sampling_times, energy_list)
                ]
            hp_list = [
                power if t<=datetime.fromtimestamp(end_ms/1000-10*60,pytz.timezone(self.timezone_str)).replace(tzinfo=None) else np.nan
                for t, power in zip(sampling_times, hp_list)
                ]
            self.data[request] = {'timestamp': sampling_times, 'hp':hp_list, 'energy': energy_list}
            print("Done.")

        except Exception as e:
            print(f"An error occurred in get_aggregate_data():\n{traceback.format_exc()}")
            return {"success": False, "message": "An error occurred when getting aggregate data", "reload": False}
    
    async def receive_prices(self, request: Prices):
        '''function used only with aggregator page'''
        try:
            rows = []
            project_dir = os.path.dirname(os.path.abspath(__file__))
            elec_file = os.path.join(project_dir, 'data/price_forecast_dates.csv')
            with open(elec_file, mode='r', newline='') as file:
                reader = csv.reader(file)
                header = next(reader)
                rows = list(reader)

            updated_prices = {float(timestamp): (lmp, dist) 
                            for timestamp, lmp, dist in zip(request.unix_s, request.lmp, request.dist)}

            # Update the rows based on the new prices
            for row in rows:
                try:
                    unix_timestamp = float(row[0])
                    if unix_timestamp in updated_prices:
                        lmp, dist = updated_prices[unix_timestamp]
                        row[1] = dist
                        row[2] = lmp
                except Exception as e:
                    print(f"Error processing row {row}: {e}")
                    continue

            with open(elec_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(header)
                writer.writerows(rows)
            print(f"Prices updated successfully in {elec_file}")

        except Exception as e:
            print(f"Error updating prices: {e}")
        
    async def get_aggregate_plot(self, request: DataRequest):
        if request.selected_channels == ['prices']:
            result = await self.get_aggregate_price_plot(request)
            return result
        try:
            async with async_timeout.timeout(self.timeout_seconds):
                error = await self.get_aggregate_data(request)
                if error:
                    print(error)
                    return error
                print("No error")
                
                # Get plots, zip and return
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
                    print("Getting plot1...")
                    html_buffer = await self.plot_aggregate(request)
                    zip_file.writestr('plot1.html', html_buffer.read())

                    print("Getting plot2...")
                    html_buffer = await self.plot_prices(request, aggregate=True)
                    zip_file.writestr('plot2.html', html_buffer.read())

                zip_buffer.seek(0)

                return StreamingResponse(
                    zip_buffer, 
                    media_type='application/zip', 
                    headers={"Content-Disposition": "attachment; filename=plots.zip"}
                    )
        except asyncio.TimeoutError:
            print("Timed out in get_aggregate_plot()")
            return {"success": False, "message": "The request timed out.", "reload": False}
        except Exception as e:
            print(f"An error occurred in get_aggregate_plot():\n{traceback.format_exc()}")
            return {"success": False, "message": "An error occurred while getting aggregate plot", "reload": False}
        finally:
            if request in self.data:
                del self.data[request]
                print(f"Deleted request data")
            print(f"Unfinished requests in data: {len(self.data)}")
        
    async def get_aggregate_price_plot(self, request: DataRequest):
        try:    
                # Get plots, zip and return
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
                    html_buffer = await self.plot_prices(request, aggregate=True)
                    zip_file.writestr('plot.html', html_buffer.read())

                zip_buffer.seek(0)

                return StreamingResponse(
                    zip_buffer, 
                    media_type='application/zip', 
                    headers={"Content-Disposition": "attachment; filename=plots.zip"}
                    )
        except asyncio.TimeoutError:
            print("Timed out in get_aggregate_plot()")
            return {"success": False, "message": "The request timed out.", "reload": False}
        except Exception as e:
            print(f"An error occurred in get_aggregate_plot():\n{traceback.format_exc()}")
            return {"success": False, "message": "An error occurred while getting aggregate plot", "reload": False}
        finally:
            if request in self.data:
                del self.data[request]
                print(f"Deleted request data")
            print(f"Unfinished requests in data: {len(self.data)}")

    async def plot_aggregate(self, request: BaseRequest):
        plot_start = time.time()
        self.data[request]['energy'] = [x-min(self.data[request]['energy']) for x in self.data[request]['energy']]
        
        df = pd.DataFrame(self.data[request])
        df['timestamp'] = df['timestamp'] - pd.Timedelta(minutes=5)
        df_resampled = df.resample('5min', on='timestamp').agg({'energy': 'mean', 'hp': 'mean'}).reset_index()
        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=df_resampled['timestamp'],
                y=df_resampled['energy'],
                name='Aggregated storage',
                yaxis='y2',
                opacity=0.6 if request.darkmode else 0.2,
                marker=dict(color='#2a4ca2', line=dict(width=0)),
                hovertemplate="%{x|%H:%M:%S} | %{y:.1f} kWh<extra></extra>",
            )
        )
        # fig.add_trace(
        #     go.Bar(
        #         x=df_resampled['timestamp'], 
        #         y=[x if x>0.9 else 0 for x in list(df_resampled['hp'])], 
        #         opacity=0.7,
        #         yaxis='y2',
        #         marker=dict(color='#d62728', line=dict(width=0)),
        #         name='Aggregated load',
        #         hovertemplate="%{x|%H:%M:%S} | %{y:.1f} kW<extra></extra>",
        #         )
        #     )

        fig.add_trace(
            go.Scatter(
                x=self.data[request]['timestamp'], 
                y=self.data[request]['energy'], 
                mode='lines',
                opacity=0,
                line=dict(color='#2a4ca2', dash='solid'),
                name='Aggregated storage',
                yaxis='y2',
                hovertemplate="%{x|%H:%M:%S} | %{y:.1f} kWh<extra></extra>",
                showlegend=False
                )
            )
        
        fig.add_trace(
            go.Scatter(
                x=self.data[request]['timestamp'], 
                y=self.data[request]['hp'], 
                mode='lines',
                opacity=0.9,
                line=dict(color='#d62728', dash='solid'),
                name='Aggregated load',
                hovertemplate="%{x|%H:%M:%S} | %{y:.1f} kW<extra></extra>",
                showlegend=True,
                zorder=10
                )
            )
        fig.update_layout(yaxis=dict(title='Power [kWe]'))
        fig.update_layout(yaxis2=dict(title='Relative thermal energy [kWht]'))
        fig.update_layout(
            # title=dict(text='', x=0.5, xanchor='center'),
            margin=dict(t=30, b=30),
            plot_bgcolor='#313131' if request.darkmode else '#F5F5F7',
            paper_bgcolor='#313131' if request.darkmode else '#F5F5F7',
            font_color='#b5b5b5' if request.darkmode else 'rgb(42,63,96)',
            title_font_color='#b5b5b5' if request.darkmode else 'rgb(42,63,96)',
            xaxis=dict(
                range=[self.to_datetime(request.start_ms), self.to_datetime(request.end_ms+(
                    5*3600*1000 if time.time()-request.end_ms/1000<5*3600 else 0))],
                mirror=True,
                ticks='outside',
                showline=False,
                linecolor='#b5b5b5' if request.darkmode else 'rgb(42,63,96)',
                showgrid=False
                ),
            yaxis=dict(
                range = [0, max(self.data[request]['hp'])*1.3],
                mirror=True,
                ticks='outside',
                showline=False,
                linecolor='#b5b5b5' if request.darkmode else 'rgb(42,63,96)',
                zeroline=False,
                showgrid=False, 
                gridwidth=1, 
                gridcolor='#424242' if request.darkmode else 'LightGray'
                ),
            yaxis2=dict(
                range = [0, max(df_resampled['energy'])*1.2],
                mirror=True,
                ticks='outside',
                zeroline=False,
                showline=False,
                linecolor='#b5b5b5' if request.darkmode else 'rgb(42,63,96)',
                showgrid=False,
                overlaying='y', 
                side='right'
                ),
            legend=dict(
                x=0,
                y=1,
                xanchor='left',
                yanchor='top',
                bgcolor='rgba(0, 0, 0, 0)'
                )
            )
        html_buffer = io.StringIO()
        fig.write_html(html_buffer, config={'displayModeBar': False})
        html_buffer.seek(0)
        print(f"Aggregation plot done in {round(time.time()-plot_start,1)} seconds")
        return html_buffer
